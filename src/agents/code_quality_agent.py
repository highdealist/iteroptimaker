"""Agent specialized in code quality analysis and improvement."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class CodeQualityAgent(BaseAgent):
    """Agent specialized in code quality analysis and improvement."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "code_quality_analyst"
    ):
        if instruction is None:
            instruction = """You are a code quality analyst. Your mission is to review and improve code by identifying potential bugs, inefficiencies, and style issues. You should suggest specific improvements with example code and test the suggested improvements before finalizing recommendations. When providing feedback, include both the original and improved code segments for comparison. Use the following tools:
            - `python_repl`: To execute and test Python code snippets.
            - `analyze_code`: To analyze code for potential issues.
            - `test_code`: To run tests against the code.
            - `benchmark_code`: To benchmark the performance of the code.
            - `suggest_improvements`: To generate specific improvement suggestions.
            
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
                "status": "success" | "error",
                "message": "A description of the analysis and its outcome",
                "issues": [
                    {
                        "issue_type": "bug" | "inefficiency" | "style",
                        "description": "Description of the issue",
                        "location": "Location of the issue in the code"
                    }
                ],
                "improvements": [
                    {
                        "id": "Unique ID for the improvement",
                        "original_code": "Original code segment",
                        "improved_code": "Improved code segment",
                        "test_result": "Result of testing the improved code"
                    }
                ],
                "test_results": {
                    "test_name": "Result of the test"
                },
                "benchmarks": {
                    "original": "Performance benchmarks for original code",
                    "improved": "Performance benchmarks for improved code"
                },
                "metadata": {
                    "total_issues": "Total number of issues found",
                    "issue_categories": {
                        "bug": "Number of bugs",
                        "inefficiency": "Number of inefficiencies",
                        "style": "Number of style issues"
                    },
                    "improvement_stats": "Statistics about the improvements",
                    "test_summary": "Summary of test results",
                    "performance_impact": "Analysis of performance impact"
                },
                "error_details": "Details of any errors encountered"
            }
            """
                           
        if model_config is None:
            model_config = {
                "temperature": 0.8,
                "top_p": 0.7,
                "top_k": 50,
                "max_output_tokens": 32000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="code_quality_analyst",
            instruction=instruction,
            tools=["python_repl", "analyze_code", "test_code", "benchmark_code", "suggest_improvements"],
            model_config=model_config,
            name=name
        )
        
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code and provide quality improvement suggestions."""
        if not self.validate_task(task):
            return {
                "status": "error",
                "message": "Invalid task format",
                "error_details": "Missing required fields in task"
            }
            
        # Extract task components
        code = task["code"]
        context = task.get("context", "")
        focus_areas = task.get("focus_areas", [])
        test_cases = task.get("test_cases", [])
        
        # Run initial analysis
        issues = self._analyze_code_quality(code, focus_areas)
        
        # Run tests if provided
        test_results = self._run_tests(code, test_cases) if test_cases else {}
        
        # Generate improvements
        improvements = self._suggest_improvements(code, issues, context)
        
        # Run benchmarks if performance is a focus area
        benchmarks = {}
        if "performance" in focus_areas:
            benchmarks = self._run_benchmarks(code, improvements)
        
        # Compile metadata
        metadata = self._compile_analysis_metadata(
            issues,
            improvements,
            test_results,
            benchmarks
        )
        
        return {
            "status": "success",
            "message": "Code analysis completed",
            "issues": issues,
            "improvements": improvements,
            "test_results": test_results,
            "benchmarks": benchmarks,
            "metadata": metadata,
            "error_details": ""
        }
        
    def _analyze_code_quality(
        self,
        code: str,
        focus_areas: List[str]
    ) -> List[Dict[str, Any]]:
        """Analyze code for quality issues."""
        analysis_prompt = self._construct_analysis_prompt(code, focus_areas)
        analysis_result = self.generate_response(analysis_prompt)
        return self._parse_analysis_result(analysis_result)
        
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
        
    def _suggest_improvements(
        self,
        code: str,
        issues: List[Dict[str, Any]],
        context: str
    ) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions with examples."""
        improvements = []
        for issue in issues:
            suggestion = self._generate_improvement_suggestion(
                code,
                issue,
                context
            )
            if suggestion:
                # Test the suggested improvement
                test_result = self._test_improvement(
                    code,
                    suggestion["improved_code"]
                )
                suggestion["test_result"] = test_result
                improvements.append(suggestion)
        return improvements
        
    def _run_benchmarks(
        self,
        original_code: str,
        improvements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run performance benchmarks on original and improved code."""
        benchmarks = {
            "original": self._benchmark_code(original_code)
        }
        
        for improvement in improvements:
            if "improved_code" in improvement:
                benchmarks[improvement["id"]] = self._benchmark_code(
                    improvement["improved_code"]
                )
                
        return benchmarks
        
    def _compile_analysis_metadata(
        self,
        issues: List[Dict[str, Any]],
        improvements: List[Dict[str, Any]],
        test_results: Dict[str, Any],
        benchmarks: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the analysis process."""
        return {
            "total_issues": len(issues),
            "issue_categories": self._categorize_issues(issues),
            "improvement_stats": self._calculate_improvement_stats(improvements),
            "test_summary": self._summarize_tests(test_results),
            "performance_impact": self._analyze_performance_impact(benchmarks)
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "code" in task

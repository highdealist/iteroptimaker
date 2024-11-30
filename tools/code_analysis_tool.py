"""
Code analysis tool implementation.
"""
from typing import Dict, Any, List
import ast
from ..base.tool import BaseTool

class CodeAnalysisTool(BaseTool):
    """Tool for analyzing Python code."""
    
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Python code for various metrics and issues.
        
        Args:
            params: Must contain 'code' key with Python code to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.validate_params(params):
            return {"error": "Missing required parameters"}
            
        code = params["code"]
        try:
            tree = ast.parse(code)
            return {
                "metrics": self._compute_metrics(tree),
                "issues": self._find_issues(tree)
            }
        except SyntaxError as e:
            return {"error": f"Syntax error in code: {str(e)}"}
            
    def _compute_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Compute code metrics."""
        metrics = {
            "num_functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
            "num_classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
            "num_imports": len([n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]),
            "complexity": self._compute_complexity(tree)
        }
        return metrics
        
    def _compute_complexity(self, tree: ast.AST) -> int:
        """Compute cyclomatic complexity."""
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler)):
                complexity += 1
        return complexity
        
    def _find_issues(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Find potential code issues."""
        issues = []
        
        # Check for bare excepts
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    "type": "bare_except",
                    "message": "Bare except clause found",
                    "line": node.lineno
                })
                
        # Check for unused imports
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for name in node.names:
                    imports[name.asname or name.name] = node.lineno
                    
        # Add more checks as needed
        return issues
        
    @property
    def tool_name(self) -> str:
        return "code_analysis"
        
    @property
    def description(self) -> str:
        return "Analyze Python code for metrics and potential issues"
        
    @property
    def required_params(self) -> List[str]:
        return ["code"]

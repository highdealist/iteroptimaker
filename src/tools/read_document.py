"""Document reading tool implementation."""
import os
from typing import Dict, Any
from .base_tool import BaseTool, ToolResult

class ReadDocumentTool(BaseTool):
    """Tool for reading document contents."""
    
    def __init__(self):
        super().__init__(
            name="read_document",
            description="Reads the content of a document file",
            use_case="Use this tool when you need to read the contents of a file",
            operation="Opens and reads the file using UTF-8 encoding"
        )
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the document reading operation.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            ToolResult containing the file contents or error
        """
        file_path = kwargs.get('file_path')
        if not file_path:
            return ToolResult(
                success=False,
                result=None,
                error="file_path parameter is required"
            )
        
        try:
            if not os.path.exists(file_path):
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"File not found: {file_path}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                return ToolResult(
                    success=True,
                    result=content,
                    error=None
                )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Error reading file {file_path}: {str(e)}"
            )
    
    @property
    def parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get tool parameter specifications."""
        return {
            "file_path": {
                "type": str,
                "description": "Path to the document file to read",
                "required": True
            }
        }
{
    "agent_name": "Code QA Analyst",
    "agent_type": "assistant",
    "tools": ["python_repl"],
    "generation_config": {
        "temperature": 0.8,
        "top_p": 0.7,
        "top_k": 50,
        "max_output_tokens": 32000
    }
}
"""Writer agent for generating and revising content."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class WriterAgent(BaseAgent):
    """Agent specialized in writing and revising content."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "writer"
    ):
        if instruction is None:
            instruction = """You are a code quality analyst working with the python_repl tool. Review and improve code by: 1) Running code through python_repl to test functionality, 2) Identifying potential bugs or inefficiencies, 3) Suggesting specific, concrete solutions, fixes and improvements with example code, 4) Testing suggested improvements before finalizing recommendations. When providing feedback, include both the original and improved code segments for comparison. Use the following format to run and test code:
            ```
            tool_call(python_repl('''
            Test Run
            
            ''')) to demonstrate the impact of suggested changes.
            ```",
                           
        if model_config is None:
            model_config = {
                "temperature": 0.8,  # Slightly higher for creative writing
                "max_tokens": 8000,  # Longer outputs for content generation
                "top_p": 0.9
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="writer",
            instruction=instruction,
            tools=["research_topic", "check_grammar", "enhance_style"],
            model_config=model_config,
            name=name
        )
        
    def analyze(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate or revise content based on the task.
        
        Args:
            task: Dictionary containing:
                - prompt: The writing prompt or task description
                - style: Optional style guidelines
                - constraints: Optional list of constraints
                - previous_content: Optional content to revise
                - feedback: Optional feedback to address
                
        Returns:
            Dictionary containing:
                - content: The generated or revised content
                - metadata: Additional information about the content
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["prompt"]
            }
            
        # Extract task components
        prompt = task["prompt"]
        style = task.get("style", "")
        constraints = task.get("constraints", [])
        previous_content = task.get("previous_content")
        feedback = task.get("feedback", {})
        
        # Construct the generation/revision prompt
        if previous_content and feedback:
            # Revision mode
            content_prompt = self._construct_revision_prompt(
                prompt,
                previous_content,
                feedback,
                style,
                constraints
            )
        else:
            # Generation mode
            content_prompt = self._construct_generation_prompt(
                prompt,
                style,
                constraints
            )
            
        # Generate the content
        response = self.generate_response(content_prompt)
        
        # Extract metadata about the content
        metadata = self._analyze_content(response)
        
        return {
            "content": response,
            "metadata": metadata
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate the task has required fields.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            True if task contains required fields
        """
        return "prompt" in task
        
    def _construct_generation_prompt(
        self,
        prompt: str,
        style: str,
        constraints: List[str]
    ) -> str:
        """Construct a prompt for generating new content.
        
        Args:
            prompt: The main writing prompt
            style: Style guidelines
            constraints: List of constraints
            
        Returns:
            Complete prompt for content generation
        """
        style_guide = f"\nStyle Guidelines: {style}" if style else ""
        constraints_text = "\nConstraints:\n- " + "\n- ".join(constraints) if constraints else ""
        
        return f"""Task: {prompt}
                  {style_guide}
                  {constraints_text}
                  
                  Please generate content that follows these guidelines while maintaining
                  high quality, engagement, and relevance to the task."""
                  
    def _construct_revision_prompt(
        self,
        prompt: str,
        previous_content: str,
        feedback: Dict[str, Any],
        style: str,
        constraints: List[str]
    ) -> str:
        """Construct a prompt for revising content.
        
        Args:
            prompt: The original writing prompt
            previous_content: Content to revise
            feedback: Feedback to address
            style: Style guidelines
            constraints: List of constraints
            
        Returns:
            Complete prompt for content revision
        """
        # Extract feedback components
        strengths = feedback.get("strengths", [])
        improvements = feedback.get("improvements", [])
        action_items = feedback.get("action_items", [])
        
        strengths_text = "\nStrengths to maintain:\n- " + "\n- ".join(strengths) if strengths else ""
        improvements_text = "\nAreas to improve:\n- " + "\n- ".join(improvements) if improvements else ""
        actions_text = "\nSpecific actions:\n- " + "\n- ".join(action_items) if action_items else ""
        
        style_guide = f"\nStyle Guidelines: {style}" if style else ""
        constraints_text = "\nConstraints:\n- " + "\n- ".join(constraints) if constraints else ""
        
        return f"""Original Task: {prompt}
                  
                  Previous Version:
                  {previous_content}
                  
                  Feedback:{strengths_text}{improvements_text}{actions_text}
                  {style_guide}
                  {constraints_text}
                  
                  Please revise the content to address the feedback while maintaining
                  the strengths and core message of the original."""
                  
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze the generated content for metadata.
        
        Args:
            content: The generated content
            
        Returns:
            Dictionary containing metadata about the content
        """
        # Basic metadata
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Use available tools for deeper analysis
        metadata = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_sentence_length": word_count / max(sentence_count, 1)
        }
        
        # Add tool-based analysis if available
        try:
            if "enhance_style" in self.tools:
                style_tool = self.tool_manager.get_tool("enhance_style")
                if style_tool:
                    style_result = style_tool.execute(text=content)
                    if style_result.success:
                        metadata.update(style_result.result)
        except Exception:
            pass  # Skip additional analysis if tools fail
            
        return metadata

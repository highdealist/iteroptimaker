"""Workflow combining content generation and critique."""
from typing import Dict, Any, Optional, List
from ..agents.critic_agent import CriticAgent
from ..agents.writer_agent import WriterAgent
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

class CritiqueWorkflow:
    """Workflow that generates content and provides iterative feedback."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        max_iterations: int = 3
    ):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.writer = WriterAgent(
            model_manager=model_manager,
            tool_manager=tool_manager
        )
        
        self.critic = CriticAgent(
            model_manager=model_manager,
            tool_manager=tool_manager
        )
        
    def run(
        self,
        task: Dict[str, Any],
        criteria: Optional[List[str]] = None,
        context: str = ""
    ) -> Dict[str, Any]:
        """Run the critique workflow.
        
        Args:
            task: Task dictionary containing:
                - prompt: The writing prompt or task description
                - style: Optional style guidelines
                - constraints: Optional constraints
            criteria: Optional specific criteria for the critic to focus on
            context: Optional additional context
            
        Returns:
            Dictionary containing:
                - final_content: The final version of the content
                - iterations: List of previous versions with feedback
                - final_assessment: Final critique of the content
        """
        iterations = []
        current_content = None
        
        for i in range(self.max_iterations):
            # Generate or revise content
            if current_content is None:
                # Initial content generation
                writer_task = {
                    "prompt": task["prompt"],
                    "style": task.get("style", ""),
                    "constraints": task.get("constraints", [])
                }
                writer_result = self.writer.analyze(writer_task)
                current_content = writer_result["content"]
            else:
                # Revise based on previous feedback
                last_feedback = iterations[-1]["feedback"]
                revision_task = {
                    "prompt": task["prompt"],
                    "previous_content": current_content,
                    "feedback": last_feedback,
                    "style": task.get("style", ""),
                    "constraints": task.get("constraints", [])
                }
                writer_result = self.writer.analyze(revision_task)
                current_content = writer_result["content"]
            
            # Get critique of current content
            critic_task = {
                "content": current_content,
                "criteria": criteria or [],
                "context": context
            }
            feedback = self.critic.analyze(critic_task)
            
            # Store this iteration
            iterations.append({
                "version": i + 1,
                "content": current_content,
                "feedback": feedback
            })
            
            # Check if we've reached satisfactory quality
            if self._is_satisfactory(feedback):
                break
                
        return {
            "final_content": current_content,
            "iterations": iterations,
            "final_assessment": iterations[-1]["feedback"]
        }
        
    def _is_satisfactory(self, feedback: Dict[str, Any]) -> bool:
        """Check if the current version meets quality standards.
        
        Args:
            feedback: Feedback dictionary from critic
            
        Returns:
            True if quality is satisfactory
        """
        # Count significant improvements needed
        num_improvements = len([
            item for item in feedback["improvements"]
            if "minor" not in item.lower()
        ])
        
        # If there are no major improvements needed, consider it satisfactory
        return num_improvements == 0

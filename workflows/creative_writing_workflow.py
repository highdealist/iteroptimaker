"""Creative writing workflow implementation using the new agent architecture."""
from typing import Dict, Any, List, Optional
from .base_workflow import BaseWorkflow
from ..agents.writer_agent import WriterAgent
from ..agents.critic_agent import CriticAgent
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

class CreativeWritingWorkflow(BaseWorkflow):
    """Workflow for creative writing tasks with brainstorming and refinement."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        max_iterations: int = 3,
        min_ideas: int = 3
    ):
        super().__init__(model_manager, tool_manager, max_iterations)
        self.min_ideas = min_ideas
        
        # Initialize agents with creative writing focus
        writer_instruction = """You are a creative writer who excels at crafting engaging narratives.
                              Focus on vivid descriptions, compelling characters, and engaging plots.
                              Adapt your style based on the genre and target audience.
                              Use literary devices effectively to enhance the writing."""
                              
        critic_instruction = """You are a literary critic who provides insightful feedback.
                              Focus on narrative structure, character development, and thematic elements.
                              Consider pacing, dialogue, and descriptive language.
                              Provide specific examples and suggestions for improvement."""
        
        self.writer = WriterAgent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            instruction=writer_instruction,
            model_config={
                "temperature": 0.9,  # Higher for more creative outputs
                "max_tokens": 8000
            }
        )
        
        self.critic = CriticAgent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            instruction=critic_instruction,
            model_config={
                "temperature": 0.7  # Lower for more focused critique
            }
        )
        
    def run(
        self,
        format: str,
        prompt: str,
        style_guide: Optional[str] = None,
        genre: Optional[str] = None,
        target_audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the creative writing workflow.
        
        Args:
            format: Type of content (e.g., "short story", "poem")
            prompt: The writing prompt
            style_guide: Optional style guidelines
            genre: Optional genre specification
            target_audience: Optional target audience
            
        Returns:
            Dictionary containing:
                - final_content: The final written piece
                - brainstorming: List of generated ideas
                - iterations: List of drafts and feedback
                - metadata: Additional information about the process
        """
        # Phase 1: Brainstorming
        brainstorm_task = {
            "prompt": f"Generate creative ideas for a {format} about: {prompt}",
            "style": f"Genre: {genre or 'any'}\nAudience: {target_audience or 'general'}",
            "constraints": [f"Generate at least {self.min_ideas} distinct ideas"]
        }
        
        brainstorm_result = self.writer.analyze(brainstorm_task)
        ideas = self._extract_ideas(brainstorm_result["content"])
        
        # Phase 2: Initial Draft
        writing_task = {
            "prompt": prompt,
            "style": self._construct_style_guide(style_guide, genre, target_audience),
            "constraints": [
                f"Format: {format}",
                "Include vivid descriptions",
                "Maintain consistent tone"
            ]
        }
        
        iterations = []
        current_content = None
        
        # Phase 3: Iterative Improvement
        for i in range(self.max_iterations):
            # Generate or revise content
            if current_content is None:
                writer_result = self.writer.analyze(writing_task)
                current_content = writer_result["content"]
            else:
                # Include previous feedback in revision
                last_feedback = iterations[-1]["feedback"]
                revision_task = {
                    **writing_task,
                    "previous_content": current_content,
                    "feedback": last_feedback
                }
                writer_result = self.writer.analyze(revision_task)
                current_content = writer_result["content"]
            
            # Get critique
            critic_task = {
                "content": current_content,
                "criteria": [
                    "narrative structure",
                    "character development",
                    "descriptive language",
                    "pacing",
                    "dialogue",
                    "thematic elements"
                ]
            }
            feedback = self.critic.analyze(critic_task)
            
            # Store iteration
            iterations.append({
                "version": i + 1,
                "content": current_content,
                "feedback": feedback,
                "metadata": writer_result.get("metadata", {})
            })
            
            # Check if quality is satisfactory
            if self._is_satisfactory(feedback):
                break
        
        return {
            "final_content": current_content,
            "brainstorming": ideas,
            "iterations": iterations,
            "metadata": {
                "total_iterations": len(iterations),
                "final_version": len(iterations),
                "genre": genre,
                "target_audience": target_audience,
                "format": format
            }
        }
    
    def _extract_ideas(self, brainstorm_content: str) -> List[str]:
        """Extract distinct ideas from brainstorming content.
        
        Args:
            brainstorm_content: Raw brainstorming output
            
        Returns:
            List of distinct ideas
        """
        # Simple extraction by splitting on newlines and filtering
        ideas = [
            line.strip("- ").strip()
            for line in brainstorm_content.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        return ideas[:self.min_ideas]  # Ensure we don't exceed minimum
        
    def _construct_style_guide(
        self,
        style_guide: Optional[str],
        genre: Optional[str],
        target_audience: Optional[str]
    ) -> str:
        """Construct a complete style guide.
        
        Args:
            style_guide: Custom style guidelines
            genre: Content genre
            target_audience: Target audience
            
        Returns:
            Complete style guide string
        """
        components = []
        if style_guide:
            components.append(f"Style Guidelines:\n{style_guide}")
        if genre:
            components.append(f"Genre: {genre}")
        if target_audience:
            components.append(f"Target Audience: {target_audience}")
            
        return "\n\n".join(components)
        
    def _is_satisfactory(self, feedback: Dict[str, Any]) -> bool:
        """Check if the current version meets quality standards.
        
        Args:
            feedback: Feedback dictionary from critic
            
        Returns:
            True if quality is satisfactory
        """
        # Count significant improvements needed
        major_improvements = len([
            item for item in feedback["improvements"]
            if not any(minor in item.lower() for minor in ["minor", "small", "slight"])
        ])
        
        # Consider it satisfactory if there are no major improvements needed
        # and there are at least some identified strengths
        return (
            major_improvements == 0 and
            len(feedback.get("strengths", [])) >= 2
        )

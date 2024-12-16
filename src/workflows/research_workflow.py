"""Research workflow implementation using the new agent architecture."""
from typing import Dict, Any, List, Optional
from .base_workflow import BaseWorkflow
from ..agents.writer_agent import WriterAgent
from ..agents.critic_agent import CriticAgent
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager

class ResearchWorkflow(BaseWorkflow):
    """Workflow for conducting research and generating reports."""
    
    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        max_iterations: int = 3,
        min_sources: int = 3
    ):
        super().__init__(model_manager, tool_manager, max_iterations)
        self.min_sources = min_sources
        
        # Initialize agents with research focus
        writer_instruction = """You are a research writer who excels at synthesizing information.
                              Focus on clear explanations, logical structure, and proper citations.
                              Maintain academic tone and objectivity.
                              Support claims with evidence and examples."""
                              
        critic_instruction = """You are a research reviewer who provides thorough feedback.
                              Focus on argument structure, evidence quality, and citation accuracy.
                              Consider methodology, analysis, and conclusions.
                              Provide specific suggestions for improvement."""
        
        self.writer = WriterAgent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            instruction=writer_instruction,
            model_config={
                "temperature": 0.7,  # Lower for more factual outputs
                "max_tokens": 8000
            }
        )
        
        self.critic = CriticAgent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            instruction=critic_instruction,
            model_config={
                "temperature": 0.6  # Lower for more focused critique
            }
        )
        
    def run(
        self,
        topic: str,
        format: str = "report",
        depth: str = "intermediate",
        style_guide: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the research workflow.
        
        Args:
            topic: Research topic or question
            format: Output format (e.g., "report", "literature_review")
            depth: Research depth ("basic", "intermediate", "advanced")
            style_guide: Optional style guidelines
            **kwargs: Additional workflow parameters
            
        Returns:
            Dictionary containing:
                - final_content: The final research document
                - sources: List of sources used
                - methodology: Research methodology used
                - iterations: List of drafts and feedback
                - metadata: Additional information about the process
        """
        # Phase 1: Initial Research
        search_task = {
            "topic": topic,
            "depth": depth,
            "min_sources": self.min_sources
        }
        
        search_tool = self.tool_manager.get_tool("web_search")
        search_results = search_tool.execute(search_task)
        
        # Phase 2: Content Organization
        outline_task = {
            "topic": topic,
            "format": format,
            "sources": search_results["sources"],
            "style": style_guide or "",
            "constraints": [
                f"Format: {format}",
                "Include clear sections",
                "Maintain logical flow"
            ]
        }
        
        outline_result = self.writer.analyze(outline_task)
        
        # Phase 3: Content Generation
        writing_task = {
            "topic": topic,
            "outline": outline_result["content"],
            "sources": search_results["sources"],
            "style": style_guide or "",
            "constraints": [
                f"Format: {format}",
                "Cite all sources",
                "Support claims with evidence"
            ]
        }
        
        iterations = []
        current_content = None
        
        # Phase 4: Iterative Improvement
        for i in range(self.max_iterations):
            # Generate or revise content
            if current_content is None:
                writer_result = self.writer.analyze(writing_task)
                current_content = writer_result["content"]
            else:
                # Include previous feedback
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
                    "argument_structure",
                    "evidence_quality",
                    "citation_accuracy",
                    "methodology",
                    "analysis",
                    "conclusions"
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
            "sources": search_results["sources"],
            "methodology": {
                "search_strategy": search_task,
                "organization": outline_task,
                "iterations": len(iterations)
            },
            "iterations": iterations,
            "metadata": {
                "topic": topic,
                "format": format,
                "depth": depth,
                "total_iterations": len(iterations),
                "final_version": len(iterations)
            }
        }
    
    def _is_satisfactory(self, feedback: Dict[str, Any]) -> bool:
        """Check if the current version meets quality standards.
        
        Args:
            feedback: Feedback dictionary from critic
            
        Returns:
            True if quality is satisfactory
        """
        # Count significant issues
        major_issues = len([
            item for item in feedback["improvements"]
            if any(critical in item.lower() for critical in [
                "missing evidence",
                "incorrect citation",
                "flawed methodology",
                "unsupported claim"
            ])
        ])
        
        # Consider it satisfactory if there are no major issues
        # and there are at least some identified strengths
        return (
            major_issues == 0 and
            len(feedback.get("strengths", [])) >= 2
        )

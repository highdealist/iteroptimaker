"""Research workflow implementation using the new agent architecture."""
from typing import Dict, Any, List, Optional
from .base_workflow import BaseWorkflow, WorkflowState
from ..agents.writer_agent import WriterAgent
from ..agents.critic_agent import CriticAgent
from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import operator

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
        
    def initialize(self, input_text: str, context: str = "") -> WorkflowState:
        """Initialize workflow with input and context."""
        return WorkflowState(input=input_text, context=context)

    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the workflow steps."""
        result = self.graph.invoke(state)
        return result

    def validate(self, state: WorkflowState) -> bool:
        """Validate workflow results."""
        return True

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
                - meta Additional information about the process
        """
        
        # Initialize the state
        state = self.initialize(topic)
        
        # Run the graph
        result = self.execute(state)
        
        # Extract the final response
        final_content = result.output
        
        # Extract metadata
        metadata = result.meta
        
        # Extract chat history
        chat_history = result.chat_history
        
        # Extract intermediate results
        intermediate_results = result.intermediate_results
        
        # Record execution
        self._record_execution(
            task={
                "topic": topic,
                "format": format,
                "depth": depth,
                "style_guide": style_guide,
                **kwargs
            },
            result={
                "final_content": final_content,
                "sources": intermediate_results.get("sources", []),
                "methodology": intermediate_results.get("methodology", {}),
                "iterations": intermediate_results.get("iterations", []),
                "metadata": metadata
            },
            chat_history=chat_history
        )
        
        return {
            "final_content": final_content,
            "sources": intermediate_results.get("sources", []),
            "methodology": intermediate_results.get("methodology", {}),
            "iterations": intermediate_results.get("iterations", []),
            "metadata": metadata
        }
    
    def _create_graph(self) -> StateGraph:
        """Create a LangGraph StateGraph."""
        
        class GraphState(TypedDict):
            """State for the LangGraph."""
            input: str
            output: str
            context: str
            intermediate_results: Dict[str, Any]
            meta: Dict[str, Any]
            chat_history: List[BaseMessage]
        
        builder = StateGraph(GraphState)
        
        def _initial_research(state):
            """Initial research phase."""
            search_task = {
                "topic": state["input"],
                "depth": "intermediate",
                "min_sources": self.min_sources
            }
            search_tool = self.tool_manager.get_tool("web_search")
            search_results = search_tool.execute(search_task)
            
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "sources": search_results["sources"]
                },
                "chat_history": state.get("chat_history", []),
                "meta": {
                    **state.get("meta", {}),
                    "search_strategy": search_task
                }
            }
        
        def _content_organization(state):
            """Content organization phase."""
            outline_task = {
                "topic": state["input"],
                "format": "report",
                "sources": state["intermediate_results"]["sources"],
                "style": "",
                "constraints": [
                    "Format: report",
                    "Include clear sections",
                    "Maintain logical flow"
                ]
            }
            outline_result = self.writer.analyze(outline_task)
            
            return {
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "outline": outline_result["content"]
                },
                "chat_history": state.get("chat_history", []),
                "meta": {
                    **state.get("meta", {}),
                    "organization": outline_task
                }
            }
        
        def _content_generation(state):
            """Content generation phase."""
            writing_task = {
                "topic": state["input"],
                "outline": state["intermediate_results"]["outline"],
                "sources": state["intermediate_results"]["sources"],
                "style": "",
                "constraints": [
                    "Format: report",
                    "Cite all sources",
                    "Support claims with evidence"
                ]
            }
            
            iterations = []
            current_content = None
            
            for i in range(self.max_iterations):
                if current_content is None:
                    writer_result = self.writer.analyze(writing_task)
                    current_content = writer_result["content"]
                else:
                    last_feedback = iterations[-1]["feedback"]
                    revision_task = {
                        **writing_task,
                        "previous_content": current_content,
                        "feedback": last_feedback
                    }
                    writer_result = self.writer.analyze(revision_task)
                    current_content = writer_result["content"]
                
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
                
                iterations.append({
                    "version": i + 1,
                    "content": current_content,
                    "feedback": feedback,
                    "metadata": writer_result.get("metadata", {})
                })
                
                if self._is_satisfactory(feedback):
                    break
            
            return {
                "output": current_content,
                "intermediate_results": {
                    **state.get("intermediate_results", {}),
                    "iterations": iterations
                },
                "chat_history": state.get("chat_history", []),
                "meta": {
                    **state.get("meta", {}),
                    "total_iterations": len(iterations),
                    "final_version": len(iterations)
                }
            }
        
        builder.add_node("initial_research", _initial_research)
        builder.add_node("content_organization", _content_organization)
        builder.add_node("content_generation", _content_generation)
        
        builder.set_entry_point("initial_research")
        builder.add_edge("initial_research", "content_organization")
        builder.add_edge("content_organization", "content_generation")
        builder.add_edge("content_generation", END)
        
        return builder.compile()
    
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

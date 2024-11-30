"""Director agent for coordinating workflow between researcher and writer agents."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class DirectorAgent(BaseAgent):
    """Agent specialized in coordinating workflow and monitoring progress."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "director"
    ):
        if instruction is None:
            instruction = """You coordinate the workflow between the researcher and writer agents. Monitor the 
                           research process and writing quality. When the researcher provides information, 
                           evaluate its completeness before passing to the writer. If more research is needed, 
                           request specific additional searches. When the writer produces content, assess if it 
                           effectively uses the research. Signal workflow completion by including 'FINAL ANSWER' 
                           when all requirements are met. Keep the team focused on the original query or task."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.3,  # Low temperature for more focused coordination
                "top_p": 0.5,
                "top_k": 30,
                "max_output_tokens": 32000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="director",
            instruction=instruction,
            tools=["evaluate_research", "assess_content", "coordinate_workflow", "monitor_progress"],
            model_config=model_config,
            name=name
        )
        
    def coordinate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate workflow between researcher and writer agents.
        
        Args:
            task: Dictionary containing:
                - query: Original query or task
                - research: Research information from researcher
                - content: Content from writer
                - workflow_state: Current state of the workflow
                - requirements: Task requirements
                
        Returns:
            Dictionary containing:
                - next_action: Next action to take
                - feedback: Feedback for agents
                - workflow_status: Current workflow status
                - completion_status: Whether requirements are met
                - metadata: Additional coordination information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["query"]
            }
            
        # Extract task components
        query = task["query"]
        research = task.get("research", {})
        content = task.get("content", "")
        workflow_state = task.get("workflow_state", {})
        requirements = task.get("requirements", [])
        
        # Evaluate current state
        state_evaluation = self._evaluate_state(
            query,
            research,
            content,
            workflow_state,
            requirements
        )
        
        # Determine next action
        next_action = self._determine_next_action(state_evaluation)
        
        # Generate feedback
        feedback = self._generate_feedback(
            state_evaluation,
            next_action
        )
        
        # Check completion status
        completion_status = self._check_completion(
            state_evaluation,
            requirements
        )
        
        # Generate metadata
        metadata = self._compile_coordination_metadata(
            state_evaluation,
            next_action,
            completion_status
        )
        
        return {
            "next_action": next_action,
            "feedback": feedback,
            "workflow_status": state_evaluation["status"],
            "completion_status": completion_status,
            "metadata": metadata
        }
        
    def _evaluate_state(
        self,
        query: str,
        research: Dict[str, Any],
        content: str,
        workflow_state: Dict[str, Any],
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Evaluate the current state of the workflow."""
        # Evaluate research completeness
        research_evaluation = self._evaluate_research_completeness(
            query,
            research,
            requirements
        )
        
        # Evaluate content quality
        content_evaluation = self._evaluate_content_quality(
            content,
            research,
            requirements
        ) if content else None
        
        # Evaluate workflow progress
        workflow_evaluation = self._evaluate_workflow_progress(
            workflow_state,
            requirements
        )
        
        return {
            "research_status": research_evaluation,
            "content_status": content_evaluation,
            "workflow_status": workflow_evaluation,
            "status": self._determine_overall_status(
                research_evaluation,
                content_evaluation,
                workflow_evaluation
            )
        }
        
    def _determine_next_action(
        self,
        state_evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the next action based on state evaluation."""
        if not state_evaluation["research_status"]["complete"]:
            return self._create_research_action(
                state_evaluation["research_status"]
            )
            
        if not state_evaluation["content_status"] or \
           not state_evaluation["content_status"]["satisfactory"]:
            return self._create_writing_action(
                state_evaluation["research_status"],
                state_evaluation["content_status"]
            )
            
        return self._create_refinement_action(state_evaluation)
        
    def _generate_feedback(
        self,
        state_evaluation: Dict[str, Any],
        next_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate feedback for researcher and writer agents."""
        return {
            "researcher": self._generate_researcher_feedback(
                state_evaluation,
                next_action
            ),
            "writer": self._generate_writer_feedback(
                state_evaluation,
                next_action
            ),
            "general": self._generate_general_feedback(
                state_evaluation,
                next_action
            )
        }
        
    def _check_completion(
        self,
        state_evaluation: Dict[str, Any],
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Check if all requirements are met."""
        requirement_status = self._check_requirements(
            state_evaluation,
            requirements
        )
        
        return {
            "complete": all(requirement_status.values()),
            "requirement_status": requirement_status,
            "missing_requirements": [
                req for req, status in requirement_status.items()
                if not status
            ]
        }
        
    def _compile_coordination_metadata(
        self,
        state_evaluation: Dict[str, Any],
        next_action: Dict[str, Any],
        completion_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the coordination process."""
        return {
            "workflow_metrics": self._calculate_workflow_metrics(
                state_evaluation
            ),
            "coordination_efficiency": self._evaluate_coordination_efficiency(
                state_evaluation,
                next_action
            ),
            "completion_analysis": self._analyze_completion_status(
                completion_status
            ),
            "workflow_summary": self._generate_workflow_summary(
                state_evaluation,
                next_action,
                completion_status
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "query" in task

"""NHR (Next Human Response) agent for coordinating agent interactions and selecting next steps."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class NHRAgent(BaseAgent):
    """Agent specialized in coordinating and selecting next steps in agent interactions."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "nhr"
    ):
        if instruction is None:
            instruction = """You are the coordinator. Based on the conversation's context / message history, 
                           please choose the next agent that should provide their input from the list below, 
                           or, make a new one by following the system_prompt after the list of existing models."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.5,  # Balanced temperature for coordination decisions
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 16000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="nhr",
            instruction=instruction,
            tools=["analyze_context", "select_agent", "evaluate_response", "coordinate_workflow"],
            model_config=model_config,
            name=name
        )
        
    def select_next_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Select the next agent to respond based on conversation context.
        
        Args:
            task: Dictionary containing:
                - context: Conversation context and message history
                - available_agents: List of available agents
                - current_agent: Currently active agent
                - task_status: Current status of the task
                
        Returns:
            Dictionary containing:
                - selected_agent: The selected next agent
                - reasoning: Explanation for the selection
                - suggested_prompt: Suggested prompt for the selected agent
                - metadata: Additional selection information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["context", "available_agents"]
            }
            
        # Extract task components
        context = task["context"]
        available_agents = task["available_agents"]
        current_agent = task.get("current_agent", None)
        task_status = task.get("task_status", {})
        
        # Analyze context
        context_analysis = self._analyze_context(
            context,
            current_agent
        )
        
        # Evaluate available agents
        agent_evaluation = self._evaluate_agents(
            available_agents,
            context_analysis,
            task_status
        )
        
        # Select next agent
        selected_agent = self._select_agent(
            agent_evaluation,
            context_analysis
        )
        
        # Generate suggested prompt
        suggested_prompt = self._generate_prompt(
            selected_agent,
            context_analysis
        )
        
        # Generate metadata
        metadata = self._compile_selection_metadata(
            context_analysis,
            agent_evaluation,
            selected_agent
        )
        
        return {
            "selected_agent": selected_agent,
            "reasoning": self._generate_reasoning(
                selected_agent,
                context_analysis
            ),
            "suggested_prompt": suggested_prompt,
            "metadata": metadata
        }
        
    def _analyze_context(
        self,
        context: str,
        current_agent: Optional[str]
    ) -> Dict[str, Any]:
        """Analyze the conversation context."""
        analysis_prompt = self._construct_analysis_prompt(
            context,
            current_agent
        )
        analysis = self.generate_response(analysis_prompt)
        return self._parse_analysis(analysis)
        
    def _evaluate_agents(
        self,
        available_agents: List[str],
        context_analysis: Dict[str, Any],
        task_status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate available agents for the next step."""
        evaluation_prompt = self._construct_evaluation_prompt(
            available_agents,
            context_analysis,
            task_status
        )
        evaluation = self.generate_response(evaluation_prompt)
        return self._parse_evaluation(evaluation)
        
    def _select_agent(
        self,
        agent_evaluation: Dict[str, Any],
        context_analysis: Dict[str, Any]
    ) -> str:
        """Select the next agent based on evaluation."""
        selection_prompt = self._construct_selection_prompt(
            agent_evaluation,
            context_analysis
        )
        selection = self.generate_response(selection_prompt)
        return self._parse_selection(selection)
        
    def _generate_prompt(
        self,
        selected_agent: str,
        context_analysis: Dict[str, Any]
    ) -> str:
        """Generate a suggested prompt for the selected agent."""
        prompt_generation = self._construct_prompt_generation(
            selected_agent,
            context_analysis
        )
        prompt = self.generate_response(prompt_generation)
        return self._format_prompt(prompt)
        
    def _generate_reasoning(
        self,
        selected_agent: str,
        context_analysis: Dict[str, Any]
    ) -> str:
        """Generate reasoning for the agent selection."""
        reasoning_prompt = self._construct_reasoning_prompt(
            selected_agent,
            context_analysis
        )
        reasoning = self.generate_response(reasoning_prompt)
        return self._format_reasoning(reasoning)
        
    def _compile_selection_metadata(
        self,
        context_analysis: Dict[str, Any],
        agent_evaluation: Dict[str, Any],
        selected_agent: str
    ) -> Dict[str, Any]:
        """Compile metadata about the selection process."""
        return {
            "context_metrics": self._calculate_context_metrics(
                context_analysis
            ),
            "agent_scores": self._calculate_agent_scores(
                agent_evaluation
            ),
            "selection_confidence": self._calculate_selection_confidence(
                selected_agent,
                agent_evaluation
            ),
            "workflow_insights": self._extract_workflow_insights(
                context_analysis,
                agent_evaluation,
                selected_agent
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return all(field in task for field in ["context", "available_agents"])

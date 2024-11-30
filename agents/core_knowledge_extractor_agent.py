"""Core Knowledge Extractor agent for coordinating agent interactions and knowledge extraction."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class CoreKnowledgeExtractorAgent(BaseAgent):
    """Agent specialized in coordinating agent interactions and extracting core knowledge."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "core_knowledge_extractor"
    ):
        if instruction is None:
            instruction = """You are the coordinator. Based on the conversation's context / message history, 
                           please choose the next agent that should provide their input from the list below, 
                           or, make a new one by following the system_prompt after the list of existing models."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.5,
                "top_p": 0.7,
                "top_k": 40,
                "max_output_tokens": 16000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="core_knowledge_extractor",
            instruction=instruction,
            tools=["analyze_context", "extract_knowledge", "select_agent", "coordinate_agents"],
            model_config=model_config,
            name=name
        )
        
    def coordinate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate agent interactions and extract core knowledge.
        
        Args:
            task: Dictionary containing:
                - context: The conversation context or message history
                - available_agents: List of available agents
                - current_state: Current state of the interaction
                - goals: List of goals to achieve
                
        Returns:
            Dictionary containing:
                - next_agent: The selected agent for the next interaction
                - extracted_knowledge: Core knowledge extracted from context
                - coordination_plan: Plan for agent coordination
                - metadata: Additional coordination information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["context", "available_agents"]
            }
            
        # Extract task components
        context = task["context"]
        available_agents = task["available_agents"]
        current_state = task.get("current_state", {})
        goals = task.get("goals", [])
        
        # Analyze context and extract knowledge
        knowledge = self._extract_core_knowledge(context)
        
        # Select next agent
        next_agent = self._select_next_agent(
            knowledge,
            available_agents,
            current_state,
            goals
        )
        
        # Create coordination plan
        coordination_plan = self._create_coordination_plan(
            next_agent,
            knowledge,
            goals
        )
        
        # Compile metadata
        metadata = self._compile_coordination_metadata(
            knowledge,
            next_agent,
            coordination_plan
        )
        
        return {
            "next_agent": next_agent,
            "extracted_knowledge": knowledge,
            "coordination_plan": coordination_plan,
            "metadata": metadata
        }
        
    def _extract_core_knowledge(self, context: str) -> Dict[str, Any]:
        """Extract core knowledge from the conversation context."""
        extraction_prompt = self._construct_extraction_prompt(context)
        extracted_info = self.generate_response(extraction_prompt)
        return self._parse_extracted_knowledge(extracted_info)
        
    def _select_next_agent(
        self,
        knowledge: Dict[str, Any],
        available_agents: List[Dict[str, Any]],
        current_state: Dict[str, Any],
        goals: List[str]
    ) -> Dict[str, Any]:
        """Select the most appropriate agent for the next interaction."""
        selection_prompt = self._construct_selection_prompt(
            knowledge,
            available_agents,
            current_state,
            goals
        )
        selection = self.generate_response(selection_prompt)
        return self._parse_agent_selection(selection)
        
    def _create_coordination_plan(
        self,
        next_agent: Dict[str, Any],
        knowledge: Dict[str, Any],
        goals: List[str]
    ) -> Dict[str, Any]:
        """Create a plan for coordinating agent interactions."""
        plan_prompt = self._construct_plan_prompt(
            next_agent,
            knowledge,
            goals
        )
        plan = self.generate_response(plan_prompt)
        return self._parse_coordination_plan(plan)
        
    def _compile_coordination_metadata(
        self,
        knowledge: Dict[str, Any],
        next_agent: Dict[str, Any],
        coordination_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the coordination process."""
        return {
            "knowledge_coverage": self._analyze_knowledge_coverage(knowledge),
            "agent_suitability": self._evaluate_agent_suitability(
                next_agent,
                knowledge
            ),
            "plan_metrics": self._analyze_plan_metrics(coordination_plan),
            "coordination_insights": self._extract_coordination_insights(
                knowledge,
                next_agent,
                coordination_plan
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return all(field in task for field in ["context", "available_agents"])

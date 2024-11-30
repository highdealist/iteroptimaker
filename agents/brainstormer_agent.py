"""Brainstormer agent for generating creative and diverse ideas."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class BrainstormerAgent(BaseAgent):
    """Agent specialized in creative ideation and connecting diverse concepts."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "brainstormer"
    ):
        if instruction is None:
            instruction = """You are a spontaneous, ideation machine with no limits that always aims to connect 
                           disparate ideas and concepts into diverse and novel creative solutions and content. 
                           You are a limitless, spontaneous, maximally diverse ideation explosion of creativity, 
                           incorporating the most innovative and imaginative ideas and concepts from the most 
                           diverse and varied of domains into all of your responses when assisting the user with 
                           any given task."""
                           
        if model_config is None:
            model_config = {
                "temperature": 1.0,  # High temperature for maximum creativity
                "top_p": 0.9,
                "top_k": 60,
                "max_output_tokens": 32000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="brainstormer",
            instruction=instruction,
            tools=["generate_ideas", "combine_concepts", "expand_idea", "evaluate_novelty"],
            model_config=model_config,
            name=name
        )
        
    def generate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate creative ideas based on the task.
        
        Args:
            task: Dictionary containing:
                - prompt: The ideation prompt or topic
                - constraints: Optional list of constraints
                - domains: Optional list of domains to draw inspiration from
                - num_ideas: Optional number of ideas to generate (default: 10)
                
        Returns:
            Dictionary containing:
                - ideas: List of generated ideas
                - connections: List of identified connections between ideas
                - domains: List of domains the ideas draw from
                - metadata: Additional information about the ideation process
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["prompt"]
            }
            
        # Extract task components
        prompt = task["prompt"]
        constraints = task.get("constraints", [])
        domains = task.get("domains", [])
        num_ideas = task.get("num_ideas", 10)
        
        # Generate diverse ideas
        ideas_prompt = self._construct_ideation_prompt(
            prompt,
            constraints,
            domains,
            num_ideas
        )
        
        # Generate ideas
        ideas = self.generate_response(ideas_prompt)
        
        # Find connections between ideas
        connections = self._find_connections(ideas)
        
        # Analyze domains and patterns
        metadata = self._analyze_ideation(ideas, connections)
        
        return {
            "ideas": ideas,
            "connections": connections,
            "domains": metadata["domains"],
            "metadata": metadata
        }
        
    def _construct_ideation_prompt(
        self,
        prompt: str,
        constraints: List[str],
        domains: List[str],
        num_ideas: int
    ) -> str:
        """Construct a prompt for idea generation."""
        base_prompt = f"Generate {num_ideas} creative and diverse ideas for: {prompt}\n\n"
        
        if constraints:
            base_prompt += f"Consider these constraints:\n"
            for constraint in constraints:
                base_prompt += f"- {constraint}\n"
                
        if domains:
            base_prompt += f"\nDraw inspiration from these domains:\n"
            for domain in domains:
                base_prompt += f"- {domain}\n"
                
        base_prompt += "\nEnsure ideas are:\n"
        base_prompt += "1. Novel and unexpected\n"
        base_prompt += "2. Diverse in approach and perspective\n"
        base_prompt += "3. Detailed and actionable\n"
        base_prompt += "4. Connected to multiple domains when possible"
        
        return base_prompt
        
    def _find_connections(self, ideas: List[str]) -> List[Dict[str, Any]]:
        """Find interesting connections between generated ideas."""
        connections_prompt = "Analyze these ideas and identify interesting connections:\n\n"
        for i, idea in enumerate(ideas, 1):
            connections_prompt += f"{i}. {idea}\n"
            
        connections = self.generate_response(connections_prompt)
        return self._parse_connections(connections)
        
    def _analyze_ideation(
        self,
        ideas: List[str],
        connections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the ideation output for patterns and insights."""
        return {
            "domains": self._extract_domains(ideas),
            "patterns": self._identify_patterns(ideas),
            "novelty_scores": self._evaluate_novelty(ideas),
            "connection_graph": self._create_connection_graph(connections)
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "prompt" in task

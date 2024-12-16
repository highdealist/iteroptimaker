"""New agent for creating and configuring new specialized agents."""
from typing import Dict, Any, List, Optional
from .base.agent import BaseAgent

class NewAgent(BaseAgent):
    """Agent specialized in creating and configuring new agents with specific capabilities."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "new"
    ):
        if instruction is None:
            instruction = """Create and configure new specialized agents based on task requirements. 
                           Define clear roles, capabilities, and interaction patterns. Ensure new agents 
                           integrate seamlessly with existing workflow and maintain consistent standards."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.4,  # Lower temperature for precise agent configuration
                "top_p": 0.6,
                "top_k": 40,
                "max_output_tokens": 16000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="new",
            instruction=instruction,
            tools=["analyze_requirements", "configure_agent", "validate_config", "generate_documentation"],
            model_config=model_config,
            name=name
        )
        
    def create_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create and configure a new specialized agent.
        
        Args:
            task: Dictionary containing:
                - requirements: Agent requirements and capabilities
                - context: Optional workflow context
                - existing_agents: Optional list of existing agents
                - constraints: Optional constraints and limitations
                
        Returns:
            Dictionary containing:
                - agent_config: Complete agent configuration
                - documentation: Agent documentation
                - integration_guide: Integration instructions
                - metadata: Additional configuration information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["requirements"]
            }
            
        # Extract task components
        requirements = task["requirements"]
        context = task.get("context", "")
        existing_agents = task.get("existing_agents", [])
        constraints = task.get("constraints", {})
        
        # Analyze requirements
        analysis = self._analyze_requirements(
            requirements,
            context,
            existing_agents
        )
        
        # Generate configuration
        config = self._generate_configuration(
            analysis,
            constraints
        )
        
        # Validate configuration
        validation = self._validate_configuration(
            config,
            requirements,
            existing_agents
        )
        
        # Generate documentation
        documentation = self._generate_documentation(
            config,
            analysis,
            validation
        )
        
        # Generate integration guide
        integration_guide = self._generate_integration_guide(
            config,
            existing_agents
        )
        
        # Generate metadata
        metadata = self._compile_configuration_metadata(
            analysis,
            config,
            validation
        )
        
        return {
            "agent_config": config,
            "documentation": documentation,
            "integration_guide": integration_guide,
            "metadata": metadata
        }
        
    def _analyze_requirements(
        self,
        requirements: Dict[str, Any],
        context: str,
        existing_agents: List[str]
    ) -> Dict[str, Any]:
        """Analyze agent requirements and context."""
        analysis_prompt = self._construct_analysis_prompt(
            requirements,
            context,
            existing_agents
        )
        analysis = self.generate_response(analysis_prompt)
        return self._parse_analysis(analysis)
        
    def _generate_configuration(
        self,
        analysis: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate agent configuration based on analysis."""
        config_prompt = self._construct_config_prompt(
            analysis,
            constraints
        )
        config = self.generate_response(config_prompt)
        return self._parse_configuration(config)
        
    def _validate_configuration(
        self,
        config: Dict[str, Any],
        requirements: Dict[str, Any],
        existing_agents: List[str]
    ) -> Dict[str, Any]:
        """Validate agent configuration against requirements."""
        validation_prompt = self._construct_validation_prompt(
            config,
            requirements,
            existing_agents
        )
        validation = self.generate_response(validation_prompt)
        return self._parse_validation(validation)
        
    def _generate_documentation(
        self,
        config: Dict[str, Any],
        analysis: Dict[str, Any],
        validation: Dict[str, Any]
    ) -> str:
        """Generate comprehensive agent documentation."""
        doc_prompt = self._construct_doc_prompt(
            config,
            analysis,
            validation
        )
        documentation = self.generate_response(doc_prompt)
        return self._format_documentation(documentation)
        
    def _generate_integration_guide(
        self,
        config: Dict[str, Any],
        existing_agents: List[str]
    ) -> str:
        """Generate guide for integrating the new agent."""
        guide_prompt = self._construct_guide_prompt(
            config,
            existing_agents
        )
        guide = self.generate_response(guide_prompt)
        return self._format_guide(guide)
        
    def _compile_configuration_metadata(
        self,
        analysis: Dict[str, Any],
        config: Dict[str, Any],
        validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the configuration process."""
        return {
            "requirement_coverage": self._calculate_coverage(
                analysis,
                config
            ),
            "configuration_metrics": self._calculate_config_metrics(
                config
            ),
            "validation_summary": self._summarize_validation(
                validation
            ),
            "integration_insights": self._extract_integration_insights(
                config,
                analysis,
                validation
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "requirements" in task

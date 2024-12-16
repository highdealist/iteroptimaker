"""Prompter agent for enhancing and refining user prompts."""
from typing import Dict, Any, List, Optional
from agent import BaseAgent

class PrompterAgent(BaseAgent):
    """Agent specialized in enhancing and refining user prompts."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "prompter"
    ):
        if instruction is None:
            instruction = """Enhance user prompts by filling in any gaps, correcting any erroneous terminology, fixing any flawed grammar, and if appropriate reorganizing and formatting it into clear, cohesive sections with their logical flow optimized for other large language / AI models to best understand and interpret.   ideas or instructions : 1. Adding relevant details and context to ensure completeness and comprehensiveness. 2. Ensuring 
                           clarity and precision. 4. Emphasizing 
                           clarity to avoid misinterpretations. Respond with 'Instructions:' followed by 
                           the enhanced prompt."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.4,  # Lower temperature for more precise prompt enhancement
                "top_p": 0.6,
                "top_k": 30,
                "max_output_tokens": 32000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="prompter",
            instruction=instruction,
            tools=["analyze_prompt", "enhance_prompt", "validate_prompt", "suggest_improvements"],
            model_config=model_config,
            name=name
        )
        
    def enhance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance and refine user prompts.
        
        Args:
            task: Dictionary containing:
                - prompt: Original user prompt
                - context: Optional additional context
                - requirements: Optional specific requirements
                - target_model: Optional target model for prompt
                
        Returns:
            Dictionary containing:
                - enhanced_prompt: The enhanced prompt
                - improvements: List of improvements made
                - suggestions: Additional suggestions
                - metadata: Additional enhancement information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["prompt"]
            }
            
        # Extract task components
        prompt = task["prompt"]
        context = task.get("context", "")
        requirements = task.get("requirements", [])
        target_model = task.get("target_model", "")
        
        # Analyze original prompt
        analysis = self._analyze_prompt(prompt, context)
        
        # Generate improvements
        improvements = self._generate_improvements(
            prompt,
            analysis,
            requirements
        )
        
        # Enhance the prompt
        enhanced_prompt = self._enhance_prompt(
            prompt,
            improvements,
            context,
            target_model
        )
        
        # Validate enhanced prompt
        validation = self._validate_enhanced_prompt(
            enhanced_prompt,
            requirements
        )
        
        # Generate suggestions
        suggestions = self._generate_additional_suggestions(
            enhanced_prompt,
            validation
        )
        
        # Generate metadata
        metadata = self._compile_enhancement_metadata(
            analysis,
            improvements,
            validation
        )
        
        return {
            "enhanced_prompt": enhanced_prompt,
            "improvements": improvements,
            "suggestions": suggestions,
            "metadata": metadata
        }
        
    def _analyze_prompt(
        self,
        prompt: str,
        context: str
    ) -> Dict[str, Any]:
        """Analyze the original prompt."""
        analysis_prompt = self._construct_analysis_prompt(prompt, context)
        analysis = self.generate_response(analysis_prompt)
        return self._parse_analysis(analysis)
        
    def _generate_improvements(
        self,
        prompt: str,
        analysis: Dict[str, Any],
        requirements: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate specific improvements for the prompt."""
        improvements_prompt = self._construct_improvements_prompt(
            prompt,
            analysis,
            requirements
        )
        improvements = self.generate_response(improvements_prompt)
        return self._parse_improvements(improvements)
        
    def _enhance_prompt(
        self,
        prompt: str,
        improvements: List[Dict[str, Any]],
        context: str,
        target_model: str
    ) -> str:
        """Enhance the prompt based on improvements."""
        enhancement_prompt = self._construct_enhancement_prompt(
            prompt,
            improvements,
            context,
            target_model
        )
        enhanced = self.generate_response(enhancement_prompt)
        return self._format_enhanced_prompt(enhanced)
        
    def _validate_enhanced_prompt(
        self,
        enhanced_prompt: str,
        requirements: List[str]
    ) -> Dict[str, Any]:
        """Validate the enhanced prompt against requirements."""
        validation_prompt = self._construct_validation_prompt(
            enhanced_prompt,
            requirements
        )
        validation = self.generate_response(validation_prompt)
        return self._parse_validation(validation)
        
    def _generate_additional_suggestions(
        self,
        enhanced_prompt: str,
        validation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate additional suggestions for prompt improvement."""
        suggestions_prompt = self._construct_suggestions_prompt(
            enhanced_prompt,
            validation
        )
        suggestions = self.generate_response(suggestions_prompt)
        return self._parse_suggestions(suggestions)
        
    def _compile_enhancement_metadata(
        self,
        analysis: Dict[str, Any],
        improvements: List[Dict[str, Any]],
        validation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile metadata about the enhancement process."""
        return {
            "clarity_metrics": self._calculate_clarity_metrics(analysis),
            "improvement_impact": self._analyze_improvement_impact(
                improvements
            ),
            "validation_summary": self._summarize_validation(validation),
            "enhancement_insights": self._extract_enhancement_insights(
                analysis,
                improvements,
                validation
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "prompt" in task

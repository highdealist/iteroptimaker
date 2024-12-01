"""Critic agent for providing constructive feedback and analysis."""
from typing import Dict, Any, List, Optional
from agent import BaseAgent

class CriticAgent(BaseAgent):
    """Agent specialized in providing constructive criticism and feedback."""
    
    def __init__(
        self,
        model_manager,
        tool_manager,
        instruction: str = None,
        model_config: Dict[str, Any] = None,
        name: str = "critic"
    ):
        if instruction is None:
            instruction = """The critic's mission is to provide constructive feedback on the latest version 
                           of the work in progress. Focus on both strengths and areas for improvement.
                           Always maintain a balanced, professional tone. Structure your feedback with clear 
                           examples and specific suggestions. Prioritize the most impactful improvements first."""
                           
        if model_config is None:
            model_config = {
                "temperature": 0.6,  # Balanced between creativity and precision
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 32000
            }
            
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="critic",
            instruction=instruction,
            tools=["analyze_content", "evaluate_quality", "suggest_improvements", "compare_versions"],
            model_config=model_config,
            name=name
        )
        
    def critique(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content and provide structured feedback.
        
        Args:
            task: Dictionary containing:
                - content: The content to analyze
                - criteria: Optional specific criteria to focus on
                - context: Optional additional context
                - previous_version: Optional previous version for comparison
                - improvement_focus: Optional specific areas to focus improvements on
                
        Returns:
            Dictionary containing:
                - strengths: List of identified strengths
                - improvements: List of suggested improvements
                - overall_assessment: General evaluation
                - action_items: Specific steps for improvement
                - version_comparison: Comparison with previous version if provided
                - metadata: Additional analysis information
        """
        if not self.validate_task(task):
            return {
                "error": "Invalid task format",
                "required_fields": ["content"]
            }
            
        # Extract task components
        content = task["content"]
        criteria = task.get("criteria", [])
        context = task.get("context", "")
        previous_version = task.get("previous_version")
        improvement_focus = task.get("improvement_focus", [])
        
        # Perform initial analysis
        analysis = self._analyze_content(content, criteria, context)
        
        # Compare with previous version if available
        version_comparison = None
        if previous_version:
            version_comparison = self._compare_versions(
                content,
                previous_version,
                criteria
            )
            
        # Generate specific improvements
        improvements = self._generate_improvements(
            content,
            analysis,
            improvement_focus
        )
        
        # Compile action items
        action_items = self._compile_action_items(
            improvements,
            version_comparison
        )
        
        # Generate metadata
        metadata = self._compile_critique_metadata(
            analysis,
            improvements,
            version_comparison
        )
        
        return {
            "strengths": analysis["strengths"],
            "improvements": improvements,
            "overall_assessment": analysis["overall_assessment"],
            "action_items": action_items,
            "version_comparison": version_comparison,
            "metadata": metadata
        }
        
    def _analyze_content(
        self,
        content: str,
        criteria: List[str],
        context: str
    ) -> Dict[str, Any]:
        """Perform detailed content analysis."""
        analysis_prompt = self._construct_analysis_prompt(
            content,
            criteria,
            context
        )
        analysis = self.generate_response(analysis_prompt)
        return self._parse_analysis(analysis)
        
    def _compare_versions(
        self,
        current: str,
        previous: str,
        criteria: List[str]
    ) -> Dict[str, Any]:
        """Compare current version with previous version."""
        comparison_prompt = self._construct_comparison_prompt(
            current,
            previous,
            criteria
        )
        comparison = self.generate_response(comparison_prompt)
        return self._parse_comparison(comparison)
        
    def _generate_improvements(
        self,
        content: str,
        analysis: Dict[str, Any],
        focus_areas: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate specific improvement suggestions."""
        improvements_prompt = self._construct_improvements_prompt(
            content,
            analysis,
            focus_areas
        )
        improvements = self.generate_response(improvements_prompt)
        return self._parse_improvements(improvements)
        
    def _compile_action_items(
        self,
        improvements: List[Dict[str, Any]],
        version_comparison: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Compile specific action items from improvements."""
        action_items = []
        
        # Convert improvements to action items
        for improvement in improvements:
            action_items.extend(
                self._improvement_to_actions(improvement)
            )
            
        # Add actions from version comparison if available
        if version_comparison:
            action_items.extend(
                self._extract_comparison_actions(version_comparison)
            )
            
        return self._prioritize_actions(action_items)
        
    def _compile_critique_metadata(
        self,
        analysis: Dict[str, Any],
        improvements: List[Dict[str, Any]],
        version_comparison: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile metadata about the critique process."""
        return {
            "analysis_metrics": self._calculate_analysis_metrics(analysis),
            "improvement_coverage": self._analyze_improvement_coverage(
                improvements,
                analysis
            ),
            "progress_metrics": self._calculate_progress_metrics(
                version_comparison
            ) if version_comparison else None,
            "critique_summary": self._generate_critique_summary(
                analysis,
                improvements,
                version_comparison
            )
        }
        
    def validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate that the task has required fields."""
        return "content" in task
        
    def _construct_analysis_prompt(
        self,
        content: str,
        criteria: List[str],
        context: str
    ) -> str:
        """Construct the analysis prompt."""
        # Implement prompt construction logic here
        pass
        
    def _construct_comparison_prompt(
        self,
        current: str,
        previous: str,
        criteria: List[str]
    ) -> str:
        """Construct the comparison prompt."""
        # Implement prompt construction logic here
        pass
        
    def _construct_improvements_prompt(
        self,
        content: str,
        analysis: Dict[str, Any],
        focus_areas: List[str]
    ) -> str:
        """Construct the improvements prompt."""
        # Implement prompt construction logic here
        pass
        
    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """Parse the analysis response."""
        # Implement parsing logic here
        pass
        
    def _parse_comparison(self, comparison: str) -> Dict[str, Any]:
        """Parse the comparison response."""
        # Implement parsing logic here
        pass
        
    def _parse_improvements(self, improvements: str) -> List[Dict[str, Any]]:
        """Parse the improvements response."""
        # Implement parsing logic here
        pass
        
    def _improvement_to_actions(
        self,
        improvement: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert an improvement to action items."""
        # Implement conversion logic here
        pass
        
    def _extract_comparison_actions(
        self,
        version_comparison: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract action items from version comparison."""
        # Implement extraction logic here
        pass
        
    def _prioritize_actions(
        self,
        action_items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prioritize action items."""
        # Implement prioritization logic here
        pass
        
    def _calculate_analysis_metrics(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate analysis metrics."""
        # Implement calculation logic here
        pass
        
    def _analyze_improvement_coverage(
        self,
        improvements: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze improvement coverage."""
        # Implement analysis logic here
        pass
        
    def _calculate_progress_metrics(
        self,
        version_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate progress metrics."""
        # Implement calculation logic here
        pass
        
    def _generate_critique_summary(
        self,
        analysis: Dict[str, Any],
        improvements: List[Dict[str, Any]],
        version_comparison: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a critique summary."""
        # Implement summary generation logic here
        pass

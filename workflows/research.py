"""Research workflow implementation."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from id8r.workflows.base_workflow import BaseWorkflow, WorkflowState
from id8r.models.model_manager import ModelManager
from id8r.tools.tool_manager import ToolManager
from id8r.agents.base.agent import BaseAgent

logger = logging.getLogger(__name__)

@dataclass
class ResearchState(WorkflowState):
    """State specific to research workflow."""
    queries: List[str] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    summary: str = ""


class ResearchWorkflow(BaseWorkflow):
    """Workflow for research tasks."""

    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        researcher_agent: BaseAgent,
        writer_agent: BaseAgent,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agents={
                "researcher": researcher_agent,
                "writer": writer_agent
            },
            config=config
        )

    def initialize(self, input_text: str, context: str = "") -> ResearchState:
        """Initialize the research workflow."""
        return ResearchState(
            input=input_text,
            context=context,
            intermediate_results={},
            metadata={
                "depth": self.config.get("depth", "moderate"),
                "focus": self.config.get("focus", "general"),
                "format": self.config.get("format", "summary")
            }
        )

    def execute(self, state: ResearchState) -> ResearchState:
        """Execute the research workflow."""
        try:
            # Generate research queries
            state.queries = self._generate_queries(state)
            logger.info(f"Generated {len(state.queries)} research queries")

            # Perform searches
            for query in state.queries:
                results = self._perform_search(query)
                state.search_results.append({
                    "query": query,
                    "results": results
                })

            # Analyze findings
            state.findings = self._analyze_findings(state)
            logger.info(f"Generated {len(state.findings)} research findings")

            # Generate summary
            state.summary = self._generate_summary(state)
            state.output = state.summary

            return state

        except Exception as e:
            logger.error(f"Error in research workflow: {e}")
            raise

    def validate(self, state: ResearchState) -> bool:
        """Validate the workflow results."""
        if not state.search_results:
            logger.error("No search results generated")
            return False

        if not state.findings:
            logger.error("No findings generated")
            return False

        if not state.summary:
            logger.error("No summary generated")
            return False

        return True

    def _generate_queries(self, state: ResearchState) -> List[str]:
        """Generate research queries from input."""
        researcher = self.get_agent("researcher")
        if not researcher:
            raise ValueError("Researcher agent not found")

        prompt = self._build_query_prompt(state)
        response = researcher.generate_response(prompt, context=state.context)
        
        # Extract queries from response
        queries = []
        for line in response.split('\n'):
            if line.strip().startswith(('-', '•', '*')):
                queries.append(line.strip().lstrip('-•* '))
        return queries

    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform search using available tools."""
        search_tool = self.tool_manager.get_tool("search")
        if not search_tool:
            raise ValueError("Search tool not found")

        try:
            results = search_tool.execute(query)
            return results
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []

    def _analyze_findings(self, state: ResearchState) -> List[str]:
        """Analyze search results and generate findings."""
        researcher = self.get_agent("researcher")
        if not researcher:
            raise ValueError("Researcher agent not found")

        findings = []
        for result in state.search_results:
            prompt = self._build_analysis_prompt(result)
            analysis = researcher.generate_response(prompt)
            findings.extend(self._extract_findings(analysis))

        return findings

    def _generate_summary(self, state: ResearchState) -> str:
        """Generate final research summary."""
        writer = self.get_agent("writer")
        if not writer:
            raise ValueError("Writer agent not found")

        prompt = self._build_summary_prompt(state)
        return writer.generate_response(prompt)

    def _build_query_prompt(self, state: ResearchState) -> str:
        """Build prompt for query generation."""
        return f"""Research Query Generation:
Topic: {state.input}
Depth: {state.metadata.get('depth')}
Focus: {state.metadata.get('focus')}

Generate a list of specific, focused research queries that will help gather comprehensive information about this topic.
Consider different aspects, perspectives, and potential areas of investigation.
Format each query as a clear, searchable question or statement.
"""

    def _build_analysis_prompt(self, search_result: Dict[str, Any]) -> str:
        """Build prompt for analyzing search results."""
        return f"""Analysis Task:
Query: {search_result['query']}

Search Results:
{chr(10).join(str(result) for result in search_result['results'])}

Analyze these results and extract key findings, insights, and relevant information.
Focus on accuracy, relevance, and potential connections between different pieces of information.
"""

    def _build_summary_prompt(self, state: ResearchState) -> str:
        """Build prompt for generating final summary."""
        findings_text = "\n".join(f"- {finding}" for finding in state.findings)
        
        return f"""Summary Generation:
Original Topic: {state.input}
Format: {state.metadata.get('format')}

Key Findings:
{findings_text}

Create a comprehensive summary that synthesizes the research findings.
Focus on:
1. Main insights and conclusions
2. Supporting evidence and data
3. Connections between different findings
4. Implications and potential applications
"""

    def _extract_findings(self, analysis: str) -> List[str]:
        """Extract structured findings from analysis text."""
        findings = []
        current_finding = []
        
        for line in analysis.split('\n'):
            line = line.strip()
            if line.startswith(('-', '•', '*')):
                if current_finding:
                    findings.append(' '.join(current_finding))
                    current_finding = []
                current_finding.append(line.lstrip('-•* '))
            elif line and current_finding:
                current_finding.append(line)
                
        if current_finding:
            findings.append(' '.join(current_finding))
            
        return findings

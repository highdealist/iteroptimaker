from .search_manager import SearchManager, initialize_search_manager
from .tool_manager import ToolManager
from ..models.model_manager import ModelManager
from ..config import GEMINI_API_KEY
from ..models.llm_providers import GeminiProvider
from ..agents.agent import AgentManager, Agent

import json
from typing import List, Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import re
import os

logger = logging.getLogger(__name__)

@dataclass
class ResearchIteration:
    query: str
    search_results: str
    summary: str
    extracted_info: Dict[str, str]
    timestamp: datetime
    confidence_level: float
    relevance_score: float
    sources: List[Dict[str, str]]

class ResearchResponse:
    def __init__(self, raw_response: str):
        self.raw_response = raw_response
        self._parse_response()
    
    def _parse_response(self):
        """Parse all sections in one pass for better efficiency."""
        # Initialize default values
        self.search_query = None
        self.extracted_info = {}
        self.final_report = None
        self.confidence_level = 0.0
        self.relevance_score = 0.0
        
        # Define all patterns
        patterns = {
            'search_query': r"SEARCH QUERY:\s*\{([^}]+)\}",
            'extracted_info': r"EXTRACTED INFO:\s*\{([^}]+)\}",
            'final_report': r"FINAL REPORT:\s*\{([^}]+)\}",
            'confidence': r"CONFIDENCE:\s*(\d*\.?\d+)",
            'relevance': r"RELEVANCE:\s*(\d*\.?\d+)"
        }
        
        # Extract all sections in one pass
        for section, pattern in patterns.items():
            match = re.search(pattern, self.raw_response, re.DOTALL)
            if match:
                if section == 'extracted_info':
                    self.extracted_info = self._parse_info_section(match.group(1))
                elif section == 'confidence':
                    self.confidence_level = float(match.group(1))
                elif section == 'relevance':
                    self.relevance_score = float(match.group(1))
                else:
                    setattr(self, section, match.group(1).strip())

    def _parse_info_section(self, info_text: str) -> Dict[str, str]:
        """Enhanced parsing of the extracted info section."""
        info_dict = {}
        current_key = None
        current_value = []
        
        for line in info_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if ':' in line and not current_key:
                key, value = line.split(':', 1)
                current_key = key.strip()
                current_value = [value.strip()]
            elif current_key and line.startswith(' '):
                current_value.append(line.strip())
            elif current_key:
                info_dict[current_key] = '\n'.join(current_value)
                if ':' in line:
                    key, value = line.split(':', 1)
                    current_key = key.strip()
                    current_value = [value.strip()]
                
        if current_key:
            info_dict[current_key] = '\n'.join(current_value)
            
        return info_dict

@dataclass
class AnalystFeedback:
    relevance_assessment: float
    direction_assessment: float
    coverage_assessment: float
    feedback: str
    recommendations: List[str]
    timestamp: datetime

@dataclass
class ResearchLog:
    researcher_output: str
    analyst_feedback: AnalystFeedback
    iteration: int
    timestamp: datetime

class ResearchAnalyst:
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.system_prompt = """You are an expert research analyst who evaluates the quality, relevance, and direction of ongoing research.
        
RESPONSE FORMAT INSTRUCTIONS:
Always structure your responses using these exact sections:

RESEARCH ASSESSMENT: {
    relevance_score: [0.0-1.0]  // How relevant is the collected information
    direction_score: [0.0-1.0]  // How well aligned with the research goal
    coverage_score: [0.0-1.0]   // How comprehensive is the coverage
    
    strengths: {
        - List key strengths of current research direction
        - Highlight particularly valuable findings
    }
    
    weaknesses: {
        - Identify gaps in coverage
        - Point out potential biases or oversights
        - Flag any irrelevant tangents
    }
    
    recommendations: {
        - Specific suggestions for next steps
        - Areas needing more focus
        - Topics to avoid or de-prioritize
    }
}

DETAILED FEEDBACK: {
    Provide specific, actionable feedback on:
    1. Information Quality
    2. Research Direction
    3. Knowledge Gaps
    4. Methodology
    5. Source Quality
}

Your task is to critically analyze research progress and provide guidance to keep the research focused and effective."""

        self.analyst = self._initialize_analyst()

    def _initialize_analyst(self) -> Agent:
        """Initialize the analyst agent."""
        try:
            agent_manager = AgentManager(tool_manager=None)  # Analyst doesn't need tools
            return agent_manager.create_agent(
                "analyst",
                "research_analyst",
                self.model_manager,
                None,
                instruction=self.system_prompt,
                model_config={
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.95
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize analyst agent: {e}")
            raise

    def analyze_research(self, topic: str, research_log: Dict[str, List[ResearchLog]], 
                        current_knowledge: Dict[str, Dict]) -> AnalystFeedback:
        """Analyze current research progress and provide feedback."""
        try:
            # Prepare analysis prompt
            analysis_prompt = self._prepare_analysis_prompt(topic, research_log, current_knowledge)
            
            # Get analyst's assessment
            response = self.analyst.generate_response(analysis_prompt)
            
            # Parse analyst's response
            feedback = self._parse_analyst_response(response)
            
            return feedback
            
        except Exception as e:
            logger.error(f"Research analysis failed: {e}")
            raise

    def _prepare_analysis_prompt(self, topic: str, research_log: Dict[str, List[ResearchLog]], 
                               current_knowledge: Dict[str, Dict]) -> str:
        """Prepare the prompt for the analyst."""
        return f"""Analyze the current research progress on topic: {topic}

Research History:
{self._format_research_log(research_log)}

Current Knowledge State:
{json.dumps(current_knowledge, indent=2)}

Provide a comprehensive analysis following the required response format."""

    def _format_research_log(self, research_log: Dict[str, List[ResearchLog]]) -> str:
        """Format research log for the prompt."""
        formatted_log = []
        for iteration, logs in research_log.items():
            formatted_log.append(f"\nIteration {iteration}:")
            for log in logs:
                formatted_log.append(f"Researcher Output: {log.researcher_output}")
                formatted_log.append(f"Previous Analysis: {log.analyst_feedback.feedback}")
        return "\n".join(formatted_log)

    def _parse_analyst_response(self, response: str) -> AnalystFeedback:
        """Parse the analyst's response into structured feedback."""
        # Extract scores using regex
        relevance_score = float(re.search(r"relevance_score:\s*(\d*\.?\d+)", response).group(1))
        direction_score = float(re.search(r"direction_score:\s*(\d*\.?\d+)", response).group(1))
        coverage_score = float(re.search(r"coverage_score:\s*(\d*\.?\d+)", response).group(1))
        
        # Extract recommendations
        recommendations_match = re.search(r"recommendations:\s*\{([^}]+)\}", response, re.DOTALL)
        recommendations = [r.strip() for r in recommendations_match.group(1).split('-') if r.strip()]
        
        # Extract detailed feedback
        feedback_match = re.search(r"DETAILED FEEDBACK:\s*\{([^}]+)\}", response, re.DOTALL)
        feedback = feedback_match.group(1).strip()
        
        return AnalystFeedback(
            relevance_assessment=relevance_score,
            direction_assessment=direction_score,
            coverage_assessment=coverage_score,
            feedback=feedback,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

class ResearchAgent:
    def __init__(self, model_manager: ModelManager, tool_manager: ToolManager):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.research_history: List[ResearchIteration] = []
        self.knowledge_base: Dict[str, Dict] = {
            'facts': {},
            'sources': {},
            'topics': {},
            'uncertainties': {},
            'metadata': {
                'last_updated': None,
                'confidence_history': [],
                'relevance_history': []
            }
        }
        
        # Load agent configuration
        with open('agents.json', encoding='utf-8', errors='ignore') as f:
            agents_config = json.load(f)
            self.researcher_config = agents_config['researcher']
        
        self.system_prompt = """You are an expert research agent that conducts thorough investigations through iterative web searches.

RESPONSE FORMAT INSTRUCTIONS:
Always structure your responses using these exact sections:

SEARCH QUERY: {
    Write a specific, focused search query based on current knowledge gaps
    Format: "keyword1 keyword2 -exclude_term site:domain.com"
}

EXTRACTED INFO: {
    key_fact1: detailed description with source reference
    key_fact2: multi-line description
               with continuation and source
    source1: {url: link, credibility: score, date: timestamp}
    uncertainty1: specific areas needing clarification
}

CONFIDENCE: [0.0-1.0]
RELEVANCE: [0.0-1.0]

FINAL REPORT: {
    Only include this section when confidence >= 0.9 or max iterations reached
    Structure the report with:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Sources and Citations
    5. Reliability Assessment
    6. Further Research Needed
}

RESEARCH PROCESS:
1. Analyze current knowledge gaps
2. Formulate precise search queries using search operators
3. Extract & organize key information with source tracking
4. Assess information reliability and relevance
5. Generate comprehensive report when ready

EVALUATION CRITERIA:
- Information relevance and reliability (scored 0-1)
- Source credibility with explicit scoring
- Knowledge completeness with gap analysis
- Logical connections between facts
- Contradictions and uncertainties tracking

Your task is to build comprehensive understanding through iterative research while maintaining strict response formatting."""
        
        self.researcher = self._initialize_researcher()
        self.research_log = {}
        self.current_iteration = 0
        self.analyst = ResearchAnalyst(model_manager)

    def _initialize_researcher(self) -> Agent:
        """Initialize the researcher agent with proper configuration."""
        try:
            agent_manager = AgentManager(tool_manager=self.tool_manager)
            return agent_manager.create_agent(
                "researcher",
                self.researcher_config['agent_type'],
                self.model_manager,
                self.tool_manager,
                instruction=self.system_prompt,
                tools=["web_search", "foia_search", "search_arxiv"],
                model_config={
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "top_p": 0.95
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize researcher agent: {e}")
            raise

    def update_knowledge_base(self, extracted_info: Dict[str, str], relevance_score: float):
        """Enhanced knowledge base update with metadata tracking."""
        current_time = datetime.now()
        
        # Update metadata
        self.knowledge_base['metadata']['last_updated'] = current_time
        self.knowledge_base['metadata']['relevance_history'].append({
            'score': relevance_score,
            'timestamp': current_time
        })
        
        # Process extracted information with improved categorization
        for key, value in extracted_info.items():
            category = self._categorize_info(key)
            if category:
                if isinstance(value, dict):
                    self.knowledge_base[category][key] = {
                        'content': value,
                        'added': current_time,
                        'relevance': relevance_score
                    }
                else:
                    self.knowledge_base[category][key] = {
                        'content': value,
                        'added': current_time,
                        'relevance': relevance_score
                    }

    def _categorize_info(self, key: str) -> Optional[str]:
        """Improved information categorization."""
        prefixes = {
            'fact_': 'facts',
            'source_': 'sources',
            'topic_': 'topics',
            'uncertainty_': 'uncertainties'
        }
        
        for prefix, category in prefixes.items():
            if key.startswith(prefix):
                return category
                
        # Try to infer category from content
        return self._infer_category(key)

    def _infer_category(self, key: str) -> str:
        """Infer the category of information based on content patterns."""
        key_lower = key.lower()
        if any(word in key_lower for word in ['url', 'link', 'source', 'reference']):
            return 'sources'
        elif any(word in key_lower for word in ['unknown', 'unclear', 'question']):
            return 'uncertainties'
        elif any(word in key_lower for word in ['topic', 'subject', 'theme']):
            return 'topics'
        return 'facts'

    def conduct_research(self, topic: str, max_iterations: int = 5, results_per_query: int = 5) -> str:
        """Conduct iterative research on a topic with analyst feedback."""
        try:
            iteration = 0
            final_report = None

            while iteration < max_iterations:
                # Generate research prompt with current knowledge state and analyst feedback
                research_prompt = self._prepare_research_prompt(topic, iteration)

                # Get structured response from researcher
                raw_response = self.researcher.generate_response(research_prompt)
                response = ResearchResponse(raw_response)

                # Get analyst feedback
                analyst_feedback = self.analyst.analyze_research(
                    topic,
                    self.research_log,
                    self.knowledge_base
                )

                # Store research log
                if iteration not in self.research_log:
                    self.research_log[iteration] = []
                
                self.research_log[iteration].append(ResearchLog(
                    researcher_output=raw_response,
                    analyst_feedback=analyst_feedback,
                    iteration=iteration,
                    timestamp=datetime.now()
                ))

                if response.search_query:
                    # Perform web search
                    try:
                        search_results = self.tool_manager.get_tool("search").execute(
                            query=response.search_query,
                            num_results=results_per_query
                        )
                        
                        # Update knowledge base with new information
                        self.update_knowledge_base(response.extracted_info, response.relevance_score)
                        
                        # Store research iteration
                        self.research_history.append(ResearchIteration(
                            query=response.search_query,
                            search_results=search_results,
                            summary=raw_response,
                            extracted_info=response.extracted_info,
                            timestamp=datetime.now(),
                            confidence_level=response.confidence_level,
                            relevance_score=response.relevance_score,
                            sources=response.sources
                        ))

                    except Exception as e:
                        logger.error(f"Search failed: {e}")
                        continue

                # Check if we should continue based on analyst feedback
                if analyst_feedback.coverage_assessment >= 0.9 and analyst_feedback.relevance_assessment >= 0.9:
                    final_report = self.generate_final_report(topic)
                    break

                iteration += 1

            # Generate final report if not already done
            if not final_report:
                final_report = self.generate_final_report(topic)

            return final_report

        except Exception as e:
            logger.error(f"Research process failed: {e}")
            raise

    def _prepare_research_prompt(self, topic: str, iteration: int) -> str:
        """Prepare research prompt with analyst feedback."""
        prompt = f"""Research Topic: {topic}

Current Knowledge Base:
{json.dumps(self.knowledge_base, indent=2)}

Research History and Analysis:
{self._format_research_history(iteration)}

Analyze the current state of research and respond in the required format with either:
1. A new search query to fill knowledge gaps, or
2. A final report if confidence level is sufficient (>=0.9)"""

        return prompt

    def _format_research_history(self, current_iteration: int) -> str:
        """Format research history with analyst feedback."""
        history = []
        for iteration in range(current_iteration):
            if iteration in self.research_log:
                for log in self.research_log[iteration]:
                    history.append(f"\nIteration {iteration}:")
                    history.append(f"Research Output: {log.researcher_output}")
                    history.append(f"Analyst Feedback: {log.analyst_feedback.feedback}")
                    history.append("Recommendations:")
                    for rec in log.analyst_feedback.recommendations:
                        history.append(f"- {rec}")
        return "\n".join(history)

    def generate_final_report(self, topic: str) -> str:
        """Generate a final report based on accumulated knowledge."""
        report_prompt = f"""Generate a comprehensive final report on '{topic}' using the accumulated knowledge:

Knowledge Base:
{json.dumps(self.knowledge_base, indent=2)}

Research History:
{json.dumps([{
    'query': iter.query,
    'summary': iter.summary,
    'confidence': iter.confidence_level
} for iter in self.research_history], indent=2)}

Provide a complete report in the FINAL REPORT format."""

        response = self.researcher.generate_response(report_prompt)
        parsed_response = ResearchResponse(response)
        return parsed_response.final_report or "Failed to generate final report"

def main():
    """Main execution function."""
    try:
        # Initialize required components
        search_manager = initialize_search_manager()
        tool_manager = ToolManager()
        model_manager = ModelManager()
        
        # Create research agent
        researcher = ResearchAgent(model_manager, tool_manager)
        
        # Get user input
        topic = input("Enter research topic: ")
        max_iterations = int(input("Enter maximum research iterations (1-5): "))
        results_per_query = int(input("Enter results per search (1-10): "))
        
        # Validate inputs
        max_iterations = min(max(1, max_iterations), 5)
        results_per_query = min(max(1, results_per_query), 10)
        
        # Conduct research and generate report
        report = researcher.conduct_research(
            topic, 
            max_iterations=max_iterations,
            results_per_query=results_per_query
        )
        
        # Print report
        print("\nFinal Research Report:")
        print("=" * 80)
        print(report)
        
    except Exception as e:
        logger.error(f"Research process failed: {e}")
        raise

if __name__ == "__main__":
    main()

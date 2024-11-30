"""Tests for the research workflow implementation."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from id8r.workflows.base_workflow import BaseWorkflow, WorkflowState
from id8r.workflows.research import ResearchWorkflow, ResearchState
from id8r.models.model_manager import ModelManager
from id8r.tools.tool_manager import ToolManager
from id8r.agents.base.agent import BaseAgent

@pytest.fixture
def mock_model_manager():
    return Mock(spec=ModelManager)

@pytest.fixture
def mock_tool_manager():
    manager = Mock(spec=ToolManager)
    manager.get_tool.return_value = Mock(execute=Mock(return_value=[{"title": "Result", "content": "Test content"}]))
    return manager

@pytest.fixture
def mock_researcher():
    agent = Mock(spec=BaseAgent)
    agent.generate_response.return_value = """
- What are the key components?
- How does it work?
- What are the benefits?
"""
    return agent

@pytest.fixture
def mock_writer():
    agent = Mock(spec=BaseAgent)
    agent.generate_response.return_value = "Summary of findings..."
    return agent

@pytest.fixture
def research_workflow(mock_model_manager, mock_tool_manager, mock_researcher, mock_writer):
    return ResearchWorkflow(
        model_manager=mock_model_manager,
        tool_manager=mock_tool_manager,
        researcher_agent=mock_researcher,
        writer_agent=mock_writer,
        config={"depth": "deep", "focus": "technical"}
    )

class TestResearchWorkflow:
    """Test suite for ResearchWorkflow."""

    def test_initialization(self, research_workflow):
        """Test workflow initialization."""
        state = research_workflow.initialize("Test topic", "Some context")
        
        assert isinstance(state, ResearchState)
        assert state.input == "Test topic"
        assert state.context == "Some context"
        assert state.metadata["depth"] == "deep"
        assert state.metadata["focus"] == "technical"

    def test_query_generation(self, research_workflow):
        """Test research query generation."""
        state = research_workflow.initialize("Test topic")
        queries = research_workflow._generate_queries(state)
        
        assert len(queries) == 3
        assert all(isinstance(q, str) for q in queries)
        assert "key components" in queries[0].lower()

    def test_search_execution(self, research_workflow):
        """Test search execution."""
        result = research_workflow._perform_search("test query")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Result"

    def test_search_tool_not_found(self, research_workflow, mock_tool_manager):
        """Test handling of missing search tool."""
        mock_tool_manager.get_tool.return_value = None
        
        with pytest.raises(ValueError, match="Search tool not found"):
            research_workflow._perform_search("test query")

    def test_search_error_handling(self, research_workflow, mock_tool_manager):
        """Test search error handling."""
        mock_tool = Mock()
        mock_tool.execute.side_effect = Exception("Search failed")
        mock_tool_manager.get_tool.return_value = mock_tool
        
        result = research_workflow._perform_search("test query")
        assert result == []

    def test_finding_analysis(self, research_workflow):
        """Test finding analysis."""
        state = ResearchState(
            input="test",
            search_results=[{
                "query": "test query",
                "results": [{"content": "test content"}]
            }]
        )
        
        findings = research_workflow._analyze_findings(state)
        assert isinstance(findings, list)
        assert len(findings) > 0

    def test_summary_generation(self, research_workflow):
        """Test summary generation."""
        state = ResearchState(
            input="test",
            findings=["Finding 1", "Finding 2"]
        )
        
        summary = research_workflow._summary_generation(state)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_full_workflow_execution(self, research_workflow):
        """Test complete workflow execution."""
        initial_state = research_workflow.initialize("Test topic")
        final_state = research_workflow.execute(initial_state)
        
        assert isinstance(final_state, ResearchState)
        assert len(final_state.queries) > 0
        assert len(final_state.search_results) > 0
        assert len(final_state.findings) > 0
        assert final_state.summary
        assert final_state.output == final_state.summary

    def test_validation(self, research_workflow):
        """Test state validation."""
        empty_state = ResearchState(input="test")
        assert not research_workflow.validate(empty_state)
        
        complete_state = ResearchState(
            input="test",
            search_results=[{"query": "q", "results": ["r"]}],
            findings=["f"],
            summary="s"
        )
        assert research_workflow.validate(complete_state)

    @pytest.mark.parametrize("missing_agent", ["researcher", "writer"])
    def test_missing_agent_handling(self, research_workflow, missing_agent):
        """Test handling of missing agents."""
        setattr(research_workflow, f"_{missing_agent}_agent", None)
        
        with pytest.raises(ValueError, match=f"{missing_agent.capitalize()} agent not found"):
            if missing_agent == "researcher":
                research_workflow._generate_queries(ResearchState(input="test"))
            else:
                research_workflow._generate_summary(ResearchState(input="test"))

    def test_extract_findings(self, research_workflow):
        """Test finding extraction from analysis text."""
        analysis = """
- Finding 1
  Additional details
- Finding 2
  More information
• Finding 3
* Finding 4
"""
        findings = research_workflow._extract_findings(analysis)
        
        assert len(findings) == 4
        assert "Finding 1 Additional details" in findings
        assert "Finding 2 More information" in findings

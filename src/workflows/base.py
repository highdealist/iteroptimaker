"""Base workflow components for the ID8R system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from ..models.model_manager import ModelManager
from ..tools.tool_manager import ToolManager
from ..agents.base.agent import BaseAgent
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, Annotated
import operator

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """Represents the current state of a workflow."""
    input: str
    context: str = ""
    intermediate_results: Dict[str, Any] = None
    output: str = ""
    metadata: Dict[str, Any] = None
    chat_history: List[BaseMessage] = None


class BaseWorkflow(ABC):
    """Abstract base class for all workflows."""

    def __init__(
        self,
        model_manager: ModelManager,
        tool_manager: ToolManager,
        agents: Dict[str, BaseAgent],
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.agents = agents
        self.config = config or {}
        self.state = None
        self.tool_node = self.tool_manager.get_tool_node()
        self.graph = self._create_graph()

    @abstractmethod
    def initialize(self, input_text: str, context: str = "") -> WorkflowState:
        """Initialize workflow with input and context."""
        pass

    @abstractmethod
    def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the workflow steps."""
        pass

    @abstractmethod
    def validate(self, state: WorkflowState) -> bool:
        """Validate workflow results."""
        pass

    def run(self, input_text: str, context: str = "") -> str:
        """Run the complete workflow."""
        try:
            # Initialize workflow
            self.state = self.initialize(input_text, context)
            logger.info(f"Initialized workflow with input: {input_text[:100]}...")

            # Execute workflow steps
            self.state = self.execute(self.state)
            logger.info("Workflow execution completed")

            # Validate results
            if not self.validate(self.state):
                logger.warning("Workflow validation failed")
                return "Error: Workflow validation failed"

            return self.state.output

        except Exception as e:
            logger.error(f"Workflow error: {e}")
            raise

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)

    def register_agent(self, name: str, agent: BaseAgent) -> None:
        """Register a new agent."""
        self.agents[name] = agent
        
    def _create_graph(self) -> StateGraph:
        """Create a LangGraph StateGraph."""
        
        class GraphState(TypedDict):
            """State for the LangGraph."""
            input: str
            context: str
            intermediate_results: Dict[str, Any]
            output: str
            metadata: Dict[str, Any]
            chat_history: List[BaseMessage]
        
        builder = StateGraph(GraphState)
        return builder

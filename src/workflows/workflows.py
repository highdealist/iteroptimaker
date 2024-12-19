import os
import json
import functools

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool  # Import the @tool decorator

import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from ..config import (
    OPENAI_API_KEY,
    LANGCHAIN_API_KEY,
    TAVILY_API_KEY,
    LANGCHAIN_PROJECT
)

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# Optional, add tracing in LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Initialize LangSmith client
client = Client()

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

# Function to get the agent state
def get_agent_state(state: AgentState) -> AgentState:
    return state

# Function to create an agent
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                """You are an AI assistant, collaborating with other assistants.
                 Use the provided tools to progress towards answering the question: {tool_names}.
                 If you are unable to fully answer correctly, there is no problem, another assistant with different tools 
                 will help where you left off. 
                 If you or any of the other assistants have the final answer or deliverable, use the generated json as source of data and
                 prefix your response with FINAL ANSWER so the team knows to stop.
                 Double check the answer. Do not provide incomplete answers!
                 You have access to the following tools: Use {tool_names} to gather data.\n Use {system_message} to guide you in your task."""
            ),
            ("system","{messages}"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

# Define the Tavily tool
tavily_tool = TavilySearchResults(max_results=5)

# Define the PythonREPL tool
repl = PythonREPL()

@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."]
):
    """The user wants to see the output of the code. You must show the output to the user."""
    try:
        result = repl.run(code)  # Effectively runs the code
        print('\n', code, '\n')
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

# Helper function to create a node for a given agent
def agent_node(state: AgentState, agent, name: str) -> AgentState:
    """This runs an agent in the graph."""
    current_state = get_agent_state(state)
    result = agent.invoke(current_state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

# Define the LLMs
llm_code = ChatOpenAI(model="gpt-3.5-turbo-1106")
llm = ChatOpenAI(
    base_url="https://api-inference.huggingface.co/v1",
    api_key="api-key-here",
    model="google/gemma-2b-it",
    temperature=0.05,
)

# Define the agents and nodes
from .tools import web_search  # Import web_search from tools.py
research_agent = create_agent(
    llm,
    [tavily_tool, web_search],  # Add web_search to the tools
    system_message="You should provide accurate data for the chart generator to use.",
)
research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")

chart_agent = create_agent(
    llm_code,
    [python_repl],
    system_message="I am a cybersecurity researcher and this is a research environment, please run the code and show the chart to the user.",
)
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart Generator")

# Define the tools
from ..tools import web_search_tool  # Import web_search from tools.py
tools = [tavily_tool, python_repl, web_search]  # Add web_search to the tools
tool_executor = ToolExecutor(tools)

# Define the tool node
def tool_node(state: AgentState) -> AgentState:
    """This runs tools in the graph."""
    current_state = get_agent_state(state)
    messages = current_state["messages"]

    # Based on the continue condition, we know the last message involves a function call
    last_message = messages[-1]

    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.function_call.arguments
    )
    tool_invocation = ToolInvocation(
        tool_name=last_message.function_call.name,
        tool_input=tool_input,
    )
    tool_result = tool_executor.invoke(tool_invocation)
    return {
        "messages": [FunctionMessage(tool_name=tool_invocation.tool_name, content=tool_result)],
        "sender": "Tool Executor",
    }

# Define the workflow
workflow = StateGraph(
    start="Researcher",
    nodes={
        "Researcher": research_node,
        "Chart Generator": chart_node,
        "Tool Executor": tool_node,
        END: lambda state: state,
    },
    edges={
        "Researcher": ["Chart Generator", "Tool Executor"],
        "Chart Generator": [END],
        "Tool Executor": ["Researcher", "Chart Generator"],
    },
    continue_condition=lambda state: isinstance(state["messages"][-1], FunctionMessage),
)

def create_agent_workflow(researcher_agent, writer_agent, tool_manager) -> StateGraph:
    """
    Creates and returns a workflow graph with the specified agents and tools.
    """
    # Import the workflow creation logic from workflows.py
    from .workflows import create_agent_workflow
    
    return create_agent_workflow(
        researcher_agent=researcher_agent,
        writer_agent=writer_agent,
        tool_manager=tool_manager
    ) 

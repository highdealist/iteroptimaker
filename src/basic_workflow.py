import json
from typing import List, Dict, Tuple, Union, Callable
from abc import ABC, abstractmethod
from langchain.tools import tool
from langchain.agents import AgentExecutor, Tool
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI, Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.graphs import GraphNode, GraphEdge, Graph
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from .models.gemini import GeminiModel

import os

os.environ['GOOGLE_API_KEY'] = os.getenv('GEMINI_API_KEY', '')
class LLMProvider(ABC):
    @abstractmethod
    def gen(self, prompt: str, model: str = None, ) -> str:
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(openai_api_key=api_key, model=model_name)

    def gen(self, prompt: str) -> str:
        return self.llm.predict(prompt)

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        self.model = GeminiModel({
            "model_name": model_name,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8096,
        })
    def gen(self, prompt: str) -> str:
        try:
            return self.model.generate(prompt)
        except Exception as e:
            fallbacks = MODEL_FALLBACKS.get("gemini", [])
            for fallback_model in fallbacks:
                try:
                    self.model = GeminiModel({
                        "model_name": fallback_model,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 8096,
                    })
                    return self.model.generate(prompt)
                except Exception as fallback_error:
                    continue
            raise Exception(f"All Gemini models failed: {str(e)}")

class OllamaProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.llm = Ollama(model=model_name)

    def gen(self, prompt: str) -> str:
        return self.llm(prompt)

class ModelFactory:
    def create_model(self, model_name: str, provider: str, api_key: str = None) -> LLMProvider:
        if provider == "openai":
            return OpenAIProvider(api_key=api_key, model_name=model_name)
        elif provider == "gemini":
            return GeminiProvider(api_key=api_key, model_name=model_name)
        elif provider == "ollama":
            return OllamaProvider(model_name=model_name)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

class ModelManager:
    def __init__(self):
        self.model_factory = ModelFactory()
        self.models: Dict[str, LLMProvider] = {}

    def get_model(self, model_name: str, provider: str, api_key: str = None) -> LLMProvider:
        if model_name not in self.models:
            self.models[model_name] = self.model_factory.create_model(model_name, provider, api_key)
        return self.models[model_name]

@tool
def web_search(query: str) -> str:
    return f"Web search results for: {query}"

@tool
def python_repl(code: str) -> str:
    return f"Python REPL output for: {code}"

class Agent:
    def __init__(self, name: str, agent_type: str, model_manager: ModelManager, tools: List[Tool], instruction: str, model_name: str, provider: str, api_key: str = None):
        self.name = name
        self.agent_type = agent_type
        self.model_manager = model_manager
        self.tools = tools
        self.instruction = instruction
        self.model_name = model_name
        self.provider = provider
        self.api_key = api_key

    def construct_prompt(self, user_input: str) -> str:
        tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        prompt = f"""
        {self.instruction}

        Available tools:
        {tool_descriptions}

        User input: {user_input}
        """
        return prompt

    def execute(self, user_input: str) -> str:
        prompt = self.construct_prompt(user_input)
        llm = self.model_manager.get_model(self.model_name, self.provider, self.api_key)
        response = llm.gen(prompt)

        if self.tool_call_detected(response):
            tool_name, tool_args = self.parse_tool_call(response)
            tool_response = self.execute_tool(tool_name, tool_args)
            response = tool_response

        return response

    def tool_call_detected(self, response: str) -> bool:
        return "Action:" in response and "Action Input:" in response

    def parse_tool_call(self, response: str) -> Tuple[str, str]:
        action_start = response.find("Action:") + len("Action:")
        action_end = response.find("\n", action_start)
        action = response[action_start:action_end].strip()

        action_input_start = response.find("Action Input:") + len("Action Input:")
        action_input = response[action_input_start:].strip()

        return action, action_input

    def execute_tool(self, tool_name: str, tool_args: str) -> str:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(tool_args)
        return f"Tool '{tool_name}' not found."

class AgentManager:
    def __init__(self, config_path: str = "agents.json"):
        self.config_path = config_path
        self.agent_configs = self._load_agent_configs()
        self.model_manager = ModelManager()

    def _load_agent_configs(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def list_agent_types(self) -> List[str]:
        return list(self.agent_configs.keys())

    def get_agent_config(self, agent_type: str) -> Dict:
        return self.agent_configs.get(agent_type)

    def create_agent(self, agent_type: str, api_key: str = None) -> Agent:
        config = self.get_agent_config(agent_type)
        if not config:
            raise ValueError(f"Agent type '{agent_type}' not found in config.")

        tools = [Tool(name=tool_name, func=globals()[tool_name], description=tool_description) for tool_name, tool_description in config.get("tools", {}).items()]

        return Agent(
            name=config.get("name", agent_type),
            agent_type=agent_type,
            model_manager=self.model_manager,
            tools=tools,
            instruction=config.get("instruction", ""),
            model_name=config.get("model_name", "gpt-3.5-turbo"),
            provider=config.get("provider", "openai"),
            api_key=api_key
        )

class WorkflowDefinition:
    def __init__(self, name: str, nodes: Dict[str, Union[Agent, Callable]], edges: List[Tuple[str, str]]):
        self.name = name
        self.nodes = nodes
        self.edges = edges
        self.graph = self._create_graph()

    def _create_graph(self) -> Graph:
        graph_nodes = {node_name: GraphNode(node_name) for node_name in self.nodes}
        graph_edges = [GraphEdge(graph_nodes[start], graph_nodes[end]) for start, end in self.edges]
        return Graph(nodes=list(graph_nodes.values()), edges=graph_edges)

    def execute(self, input: str) -> str:
        current_node_name = self.edges[0][0]
        current_input = input

        while current_node_name:
            current_node = self.nodes[current_node_name]

            if isinstance(current_node, Agent):
                response = current_node.execute(current_input)
                current_input = response
            elif callable(current_node):
                response = current_node(current_input)
                current_input = response
            else:
                raise ValueError(f"Invalid node type: {type(current_node)}")

            current_node_name = self._get_next_node(current_node_name)

        return current_input

    def _get_next_node(self, current_node_name: str) -> Union[str, None]:
        for start, end in self.edges:
            if start == current_node_name:
                return end
        return None

class WorkflowManager:
    def __init__(self, config_path: str = "workflows.json"):
        self.config_path = config_path
        self.workflow_configs = self._load_workflow_configs()
        self.agent_manager = AgentManager()

    def _load_workflow_configs(self) -> Dict:
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def list_workflow_names(self) -> List[str]:
        return list(self.workflow_configs.keys())

    def get_workflow_config(self, workflow_name: str) -> Dict:
        return self.workflow_configs.get(workflow_name)

    def create_workflow(self, workflow_name: str, api_key: str = None) -> WorkflowDefinition:
        config = self.get_workflow_config(workflow_name)
        if not config:
            raise ValueError(f"Workflow '{workflow_name}' not found in config.")

        nodes = {}
        for node_name, node_config in config.get("nodes", {}).items():
            if node_config.get("type") == "agent":
                agent_type = node_config.get("agent_type")
                nodes[node_name] = self.agent_manager.create_agent(agent_type, api_key)
            elif node_config.get("type") == "tool":
                tool_name = node_config.get("tool_name")
                nodes[node_name] = globals()[tool_name]
            else:
                raise ValueError(f"Invalid node type: {node_config.get('type')}")

        edges = config.get("edges", [])
        return WorkflowDefinition(name=workflow_name, nodes=nodes, edges=edges)

if __name__ == "__main__":
    agent_config = {
        "research_agent": {
            "name": "Research Agent",
            "instruction": "You are a research assistant. Use the web_search tool to find information.",
            "tools": {
                "web_search": "Searches the web for the given query."
            },
            "provider": "openai"
        },
        "code_agent": {
            "name": "Code Agent",
            "instruction": "You are a code execution assistant. Use the python_repl tool to execute code.",
            "tools": {
                "python_repl": "Executes the given Python code and returns the output."
            },
            "model_name": "llama2",
            "provider": "ollama"
        }
    }

    with open("agents.json", "w") as f:
        json.dump(agent_config, f, indent=4)

    workflow_config = {
        "research_and_code": {
            "nodes": {
                "research": {
                    "type": "agent",
                    "agent_type": "research_agent"
                },
                "code": {
                    "type": "agent",
                    "agent_type": "code_agent"
                }
            },
            "edges": [
                ("research", "code")
            ]
        }
    }

    with open("workflows.json", "w") as f:
        json.dump(workflow_config, f, indent=4)

    agent_manager = AgentManager()
    workflow_manager = WorkflowManager()

    workflow = workflow_manager.create_workflow("research_and_code", api_key="YOUR_OPENAI_API_KEY")

    user_input = "Find the current weather in London and then print 'Hello World' in python"
    result = workflow.execute(user_input)
    print(f"Workflow Result: {result}")




















#```markdown

# AI Assistant - README.md



## Overview



This project implements an AI assistant application that empowers users to interact with powerful AI models and tools through a user-friendly Tkinter GUI. The application allows users to define and execute complex workflows, manage AI agents, select from various Large Language Models (LLMs), and integrate custom tools. It leverages the capabilities of LangChain and LangGraph for workflow management and tool integration, and supports LLMs like Gemini, Ollama, and OpenAI's models.



## Features

Workflow Management: Create, load, save, and execute workflows defined using LangGraph. Workflows specify a sequence of actions performed by agents and tools.
Agent Management: Create, edit, and delete AI agents. Each agent possesses unique instructions, a set of tools, and a configurable LLM.
Model Selection: Choose from different LLM providers (Gemini, Ollama, OpenAI) to power each agent.
Tool Integration: Seamlessly integrate custom tools using LangChain's `@tool` decorator. Tools can range from web search functionalities to Python REPL execution.
Intuitive GUI (Tkinter): A user-friendly interface for workflow selection, input provision, agent management, model selection, and tool configuration.
Search Functionality (Future):  Plans to integrate search capabilities using various search providers like Google Custom Search, DuckDuckGo, and Brave.
Vector Database Integration (Future):  Future plans to utilize vector databases (Pinecone, Weaviate) to enable Retrieval-Augmented Generation (RAG) capabilities.



## Project Structure



#```

ai_assistant/

├── agents
│   ├── __init__.p
│   ├── agent.py          # Defines the Agent clas
│   ├── agent_manager.py  # Manages agent creation, loading, saving, and deletio
│   └── tools.py          # Defines external tools using LangChain's @too
├── models
│   ├── __init__.p
│   ├── model_manager.py  # Handles LLM selection and instantiatio
│   └── model_factory.py  # Factory pattern for creating LLM instance
├── workflows
│   ├── __init__.p
│   ├── workflow_manager.py # Manages workflow creation, loading, saving, and executio
│   └── workflow_definition.py  # Defines workflow structure using LangGrap
├── search_manager
│   ├── __init__.p
│   ├── search_provider.py    # Abstract base class for search provider
│   ├── search_api.py         # Implements specific search API
│   └── search_manager.py     # Manages searches across multiple provider
├── llm_providers
│   ├── __init__.p
│   ├── openai_provider.py   # Interface for OpenAI AP
│   ├── gemini_provider.py   # Interface for Gemini AP
│   └── ollama_provider.py   # Interface for Ollam
├── vector_database
│   ├── __init__.p
│   ├── database_manager.py  # Manages vector database interaction
│   └── embedding_service.py # Generates text embedding
├── gui
│   ├── __init__.p
│   ├── app.py           # Main Tkinter application clas
│   └── widgets.py       # Custom Tkinter widget
├── config.py            # Configuration settings (API keys, model names, etc.
├── __init__.p
└── main.py              # Main application entry poin
```



## Core Modules



### Agents

`agent.py`: Defines the `Agent` class, representing an AI agent with its instructions, tools, and model configuration.
`agent_manager.py`: Manages the lifecycle of agents, including creation, loading, saving, and deletion.
`tools.py`: Contains definitions for external tools. Each tool is a function decorated with `@tool` from LangChain, making it easily usable by agents.



### Models

`model_manager.py`: Handles the selection and instantiation of LLMs (Gemini, Ollama, OpenAI) based on agent configuration or specific requirements.
`model_factory.py`: Implements a factory pattern for creating LLM instances, potentially with caching for improved efficiency.



### Workflows

`workflow_manager.py`: Manages the lifecycle of workflows, including creation, loading, saving, and execution.
`workflow_definition.py`: Defines the structure of workflows using LangGraph. Workflows consist of "nodes" (agents or functions) and "edges" (transitions between nodes).



### Search Manager (Future Implementation)

`search_provider.py`: Defines an abstract base class for various search providers (e.g., Google Custom Search, DuckDuckGo).
`search_api.py`: Implements specific search APIs, handling rate limiting and quota management.
`search_manager.py`: Manages searches across multiple APIs and providers, potentially with caching for efficiency.



### LLM Providers

`gemini_provider.py`: Provides an interface for interacting with Google's Gemini API.
`ollama_provider.py`: Provides an interface for interacting with locally run models through Ollama.
`openai_provider.py`: Provides an interface for interacting with OpenAI's models.



### Vector Database (Future Implementation)

`database_manager.py`: Manages the creation, loading, saving, and querying of vector databases (e.g., Pinecone, Weaviate).
`embedding_service.py`: Provides methods for generating text embeddings using a chosen embedding model.



### GUI

`gui/app.py`: Contains the main Tkinter application class, responsible for UI setup, user interaction handling, and workflow orchestration.
`gui/widgets.py`: Defines custom Tkinter widgets or helper functions for creating UI elements.



### Configuration

`config.py`: Stores API keys, model names, database connection details, and other configuration settings.



### Main Entry Point

`main.py`: Initializes the application, loads configuration settings, creates the GUI, and starts the main event loop.



## Installation



1. Clone the repository:



#    ```bash

    git clone <repository_url>

    cd ai_assistant

#    ```



2. Create a virtual environment (recommended):



#    ```bash

    python3 -m venv .venv

    source .venv/bin/activate  # On Linux/macOS

#    .venv\Scripts\activate  # On Windows

#    ```



3. Install dependencies:



#    ```bash

    pip install -r requirements.txt

#    ```



#    Note: The `requirements.txt` file should contain the following:



#    ```

    langchain

    langgraph

    ollama

    google-generativeai # For Gemini API

    openai

    # Vector database libraries (if needed)

    # ... other dependencies ...

#    ```



## Usage



1. Configuration:

#    *   Update `config.py` with your API keys (Gemini, OpenAI, etc.), model names, and other necessary settings.



2. Run the application:



#    ```bash

    python main.py

#    ```



3. Interact with the GUI:

#    *   Create, edit, and delete agents.

#    *   Define and execute workflows.

#    *   Provide input to workflows.

#    *   Select LLMs for agents.

#    *   Configure tools.



## Code Examples



### Tool Integration with LangChain



#```python

# In agents/tools.py

from langchain.tools import tool



#@tool

def web_search(query: str) -> str:

#    """Searches the web for the given query and returns the results."""

    # Implement your web search logic here

    # ... e.g., using a search API ...

    return "Search results for: " + query



#@tool

def python_repl(code: str) -> str:

#    """Executes the given Python code and returns the output."""

    try:

        result = exec(code)

        if result is not None:

            return str(result)

        else:

            return "Code executed successfully."

    except Exception as e:

        return f"Error: {e}"

#```



### Example Workflow Definition



#```python

# In workflows/workflow_definition.py

from langchain_core.tools import Tool

from agents.agent import Agent

from models.model_manager import ModelManager

from agents.tools import web_search, python_repl

from typing import List, Tuple, Dict, Union



class WorkflowDefinition:

    def __init__(self, name: str, nodes: List[Union[Agent, Tool]], edges: List[Tuple[str, str]]):

        self.name = name

        self.nodes = {node.name: node for node in nodes}

        self.edges = edges



    def execute(self, input: str) -> str:

#        """Executes the workflow."""

        current_node_name = self.edges[0][0]  # Start with the first node

        current_input = input



        while True:

            current_node = self.nodes[current_node_name]



            if isinstance(current_node, Agent):

                current_input = current_node.execute(current_input)

            elif isinstance(current_node, Tool):

                current_input = current_node.run(current_input)



            next_node_name = self._get_next_node(current_node_name)

            if next_node_name is None:

                break  # End of the workflow



            current_node_name = next_node_name



        return current_input



    def _get_next_node(self, current_node_name: str) -> str:

#        """Determines the next node based on the edges."""

        for edge in self.edges:

            if edge[0] == current_node_name:

                return edge[1]

        return None  # No outgoing edge found



# Example workflow

workflow = WorkflowDefinition(

    name="Research and Summarize",

    nodes=[

        Agent(

            name="Researcher",

            agent_type="researcher",

            model_manager=ModelManager(),

            tool_manager=ToolManager(),

            instruction="You are a researcher. Use web_search to find information about...",

            tools=["web_search"],

#        ),

        Agent(

            name="Summarizer",

            agent_type="summarizer",

            model_manager=ModelManager(),

            tool_manager=ToolManager(),

            instruction="You are a summarizer. Summarize the information provided by the researcher.",

            tools=[],

#        ),

        python_repl,  # Directly add the Python REPL tool

#    ],

    edges=[

#        ("Researcher", "Summarizer"),

#        ("Summarizer", "python_repl"),

#    ],

#)



# Execute the workflow

result = workflow.execute(input="What is the capital of France?")

print(result)

#```



### Agent Execution (Simplified)



#```python

# In agents/agent.py

class Agent:

    def __init__(self, name, agent_type, model_manager, tool_manager, instruction, tools):

        self.name = name

        self.agent_type = agent_type

        self.model_manager = model_manager

        self.tool_manager = tool_manager

        self.instruction = instruction

        self.tools = tools

        self.model_name = None # to be set by user via GUI or config



    def execute(self, user_input):

        prompt = self.construct_prompt(self.instruction, self.tools, user_input)

        response = self.model_manager.get_model(self.model_name).generate_text(prompt)

        if self.tool_call_detected(response):

            tool_name, tool_args = self.parse_tool_call(response)

            tool_response = self.tool_manager.execute_tool(tool_name, tool_args)

            response = tool_response # In a real scenario you might want to feed this back to the LLM

        return response



    def construct_prompt(self, instruction, tools, user_input):

        # Create a detailed prompt here for the LLM that includes instructions,

        # tool descriptions, and the user input.

        # ...

        pass



    def tool_call_detected(self, response):

        # Implement logic to detect if the LLM is requesting to use a tool.

        # This might involve parsing the response for a specific format or keyword.

        # ...

        pass



    def parse_tool_call(self, response):

        # Implement logic to extract the tool name and arguments from the LLM's response.

        # ...

        pass

#```



## Algorithms and Pseudocode



### Workflow Execution



#```

FUNCTION execute_workflow(workflow, user_input):

    FOR node IN workflow.nodes:

        IF node IS Agent:

            response = node.execute(user_input)  // Agent interacts with LLM and tools

            user_input = response // Pass agent's output to the next node

        ELSE IF node IS Tool:

            response = node(user_input) // Execute the tool directly

            user_input = response // Pass tool's output to the next node

    RETURN user_input

#```



### Agent Execution (Detailed)



#```

CLASS Agent:

    FUNCTION execute(user_input):

        prompt = construct_prompt(self.instructions, self.tools, user_input)

        response = self.model_manager.get_model(self.model_name).generate_text(prompt)

        IF tool_call_detected(response):

            tool_name, tool_args = parse_tool_call(response)

            tool_response = self.tool_manager.execute_tool(tool_name, tool_args)

            response = tool_response // Potentially feed this back to the LLM for further processing

        RETURN response

#```



## Technical Considerations

Scalability: The modular design allows for scaling individual components. Asynchronous operations and caching should be used for improved performance.
Security: Store API keys and sensitive data securely using environment variables or dedicated secrets management solutions. Implement input validation to prevent injection attacks.
Performance: Utilize asynchronous programming (e.g., `asyncio`) for I/O-bound operations. Implement caching to minimize redundant computations.
Error Handling: Implement comprehensive error handling to gracefully manage API errors, network issues, and unexpected inputs.
Dependency Management: Use a virtual environment and a `requirements.txt` file to manage project dependencies effectively.



## Contributing



Contributions to this project are welcome! Please follow these steps to contribute:



1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Make your changes and commit them with clear commit messages.

4. Push your branch to your forked repository.

5. Submit a pull request to the main repository.



## License



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

#```


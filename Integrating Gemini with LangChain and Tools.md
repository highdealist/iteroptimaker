# Integrating Gemini with LangChain and Tools

## Overview

This documentation focuses on integrating Google's Gemini LLM with LangChain and custom tools.

Installation
To use LangChain and Gemini, install the necessary packages:

`pip install langchain langchain-gemini`

###### Configuration

Set up your API key and other necessary settings with environment variables or using a configuration file.

Environment Variables:

```
export GEMINI_API_KEY=your_api_key
export GEMINI_API_URL=https://api.gemini.com
```

Configuration File:

config.py

```
#config.py
GEMINI_API_KEY = 'your_api_key'
GEMINI_API_URL = 'https://api.gemini.com'
```

Core Functionality

### Initialize the Langchain GeminiLLM Class and the Gemini Provider

First, you need to initialize the Gemini provider using the GeminiLLM class from langchain-gemini.

```
from langchain.llms import GeminiLLM
gemini = GeminiLLM(api_key=GEMINI_API_KEY, api_url=GEMINI_API_URL)
```

### Generate content

```
response = gemini.generate("Tell me about Large Language Models.")
print(response)
```

### Chat Interactions

The chat method is used to create chat interactions with the model.

# Chat with the model

response = gemini.chat(
    messages=[
        {
            'role': 'user',
            'content': 'Please summarize this article:\n\n' + article,
        },
    ]
)

print(response['message']['content'])
Tool Integration
LangChain provides a @tool decorator to define custom tools. These tools can be used by agents to perform specific tasks.

Define a Custom Tool
from langchain.tools import tool

@tool
def web_search(query: str) -> str:
    """Perform a web search using a specified query."""
    # Implement your web search logic here
    return f"Search results for: {query}"
Use the Tool in an Agent
You can integrate the custom tool into an agent and use it in a workflow.

from langchain.agents import Agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

# Define the prompt template

template = """Question: {question}
Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the agent with the Gemini model and custom tool

agent = Agent(llm=gemini, tools=[web_search])

# Create the agent executor

agent_executor = AgentExecutor(agent=agent, prompt=prompt)

# Invoke the agent

response = agent_executor.invoke({"question": "What is LangChain?"})

print(response)
Workflow Management
LangChain uses LangGraph to define and manage workflows. A workflow consists of nodes (agents or functions) and edges (transitions between nodes).

Define a Workflow
from langchain.graph import LangGraph

# Define the workflow

graph = LangGraph()

# Add nodes (agents or functions)

graph.add_node("start", agent=agent)
graph.add_node("search", tool=web_search)
graph.add_node("end", tool=lambda x: f"Final result: {x}")

# Add edges (transitions between nodes)

graph.add_edge("start", "search")
graph.add_edge("search", "end")

# Execute the workflow

result = graph.execute("start", {"question": "What is LangChain?"})

print(result)
Error Handling
LangChain provides robust error handling. When an error occurs, it raises a LangChainError with a descriptive message. You can catch this error and handle it appropriately.

from langchain import LangChainError

try:
    # Generate content
    response = gemini.generate("Hello, Gemini. What's the weather like in a non-existent city?")
    print(response)
except LangChainError as e:
    print(f"Error: {e}")

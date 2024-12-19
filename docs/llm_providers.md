    

    

    

    

    # Ollama Cheat Sheet for Python

**Ollama is a tool for running large language models locally. This document focuses on using Ollama with Python and the python libraries langchain, langgraph, and .**

**Core Functionality:**

## **Creating an Ollama Client:**

```
import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate(self, model, prompt, **kwargs):
        url = f"{self.base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, **kwargs}
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
```

content_copy**Use code** [with caution](https://support.google.com/legal/answer/13505487).**Python**

* **Generating Text:**

```
client = OllamaClient()

response = client.generate(
    model="mistral",  # Replace with your model name
    prompt="Explain quantum computing in simple terms.",
    stream=True # If you want to stream the response
)

if response.get("stream"): # Handle streaming responses
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            print(chunk)

elif response.get("response"): # Handle non streaming response
    print(response["response"])
```

content_copy**Use code** [with caution](https://support.google.com/legal/answer/13505487).**Python**

* **Listing Available Models:**

```
client = OllamaClient()

response = client.get("/api/models")
models = response.json()["models"]
for model in models:
    print(model["name"])
```

content_copy**Use code** [with caution](https://support.google.com/legal/answer/13505487).**Python**

**Example incorporating LangGraph:**

```
from langgraph import StateGraph
from langchain.tools import DuckDuckGoSearchRun

# Assuming OllamaClient from previous examples

def get_search_query(state):
    state["query"] = input("Enter your search query: ")
    return state

def search_web(state):
    search = DuckDuckGoSearchRun()
    state["results"] = search.run(state["query"])
    return state

def run_ollama(state):
    client = OllamaClient()
    response = client.generate(
        model="mistral",
        prompt=f"Summarize the following search results: {state['results']}"
    )
    state["summary"] = response["response"]
    return state

def process_results(state):
    print(state["summary"])
    return state

# Create a StateGraph
workflow = StateGraph()

# Add nodes and edges (similar to LangGraph example)
# ...

# Execute the workflow
workflow.run()
```

content_copy**Use code** [with caution](https://support.google.com/legal/answer/13505487).**Python**

**This cheat sheet provides a basic overview and examples for using Ollama with Python. Consult the Ollama documentation for more advanced features and options. This information is current as of December 11, 2024, but may change with future updates to Ollama.**

# Integrating Multiple LLM providers into one application

### llm_provider.py

```py
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def initialize_model(self, api_key, model_name, temperature):
        pass

    @abstractmethod
    def generate_response(self, model, prompt):
        pass
```

### ollama_provider.py

```py
from llm_provider import LLMProvider
from langchain.llms import Ollamaclass OllamaProvider(LLMProvider):
    def initialize_model(self, model_name="llama3.1", temperature=0):
        # Ollama does not require an API key
        return Ollama(base_url="http://localhost:11434", model=model_name, temperature=temperature)  
	def generate_response(self, model, prompt):
        return model(prompt)
```

### openrouter_provider.py

```
from.llm_provider import LLMProvider
from langchain.llms import OpenRouterLLM

class OpenRouterProvider(LLMProvider):
    def initialize_model(self, api_key, model_name="j2-ultra", temperature=0):
        return OpenRouterLLM(api_key=api_key, model=model_name, temperature=temperature)

    def generate_response(self, model, prompt):
        return model(prompt)
```

### providers/gemini_provider.py

```
from llm_provider import LLMProvider
from langchain_gemini import ChatGemini

class GeminiProvider(LLMProvider):
    def initialize_model(self, api_key, model_name, temperature):
        return ChatGemini(api_key=api_key, model=model_name, temperature=temperature)

    def generate_response(self, model, prompt):
        return model(prompt)
```

## groq_provider.py

from langchain_groq import ChatGroq

class GroqProvider(LLMProvider):
    def initialize_model(self, api_key, 	model_name="mixtral-8x7b-32768", temperature=0):
        return ChatGroq(api_key=api_key, model=model_name, temperature=temperature)

    def generate_response(self, model, prompt):
        return model(prompt)

### Ensure parent class for providers has a universal interface

Ensure that the chatbot_function in chatbot.py is generic enough to handle different providers:

```
from .providers.groq_provider import GroqProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .prompts.chat_prompt import create_prompt

def chatbot_function(provider, api_key, system_message, human_message):
    model = provider.initialize_model(api_key)
    prompt = create_prompt(system_message, human_message)
    chain = prompt | model
    return provider.generate_response(model, prompt)

```

4. Update Main Application
   Update main.py to include the Ollama provider:

```
from .chatbot.chatbot import chatbot_function
from .providers.groq_provider import GroqProvider
from .providers.gemini_provider import GeminiProvider
from .providers.ollama_provider import OllamaProvider
from .tools.tool_integration import create_tool_node
from .graph.graph_builder import build_conversation_graphdef main():
    groq_api_key = "your_groq_api_key"
    gemini_api_key = "your_gemini_api_key"
    system_message = "You are a helpful assistant."
    human_message = "Explain the importance of low latency LLMs."    providers = {
        "groq": GroqProvider(),
        "gemini": GeminiProvider(),
        "ollama": OllamaProvider()
    }   for provider_name, provider in providers.items():
        if provider_name == "ollama":
            chatbot_func = lambda state: chatbot_function(provider, None, system_message, human_message)
        else:
            chatbot_func = lambda state: chatbot_function(provider, groq_api_key if provider_name == "groq" else gemini_api_key, system_message, human_message)  
	tool_node = create_tool_node(wikipedia_tool)
        graph = build_conversation_graph(chatbot_func, tool_node)
        graph.start()if name == "main":
    main()
```

Explanation
OllamaProvider Class: This class inherits from LLMProvider and implements the initialize_model and generate_response methods specific to Ollama. Notably, Ollama does not require an API key, so we omit it in the initialize_model method.

Chatbot Function: This function is generic and can handle any provider that adheres to the LLMProvider interface. It initializes the model, creates a prompt, and generates a response.

Main Application: The main function now includes all three providers: Groq, Gemini, and Ollama. It iterates through each provider, sets up the chatbot function accordingly, and starts the conversation graph.

By following these steps, we have successfully integrated Ollama into our multi-provider chatbot system, allowing for flexible usage of different LLM providers based on the requirements and availability of models.

To integrate Ollama with LangChain and LangGraph using a localhost server and Python client, we need to follow these steps:

Install Ollama on the local machine and start the Ollama server using ollama serve.
Create a new file ollama_provider.py in the providers directory, which defines the OllamaProvider class that inherits from LLMProvider.
Implement the initialize_model and generate_response methods in ollama_provider.py to interact with the Ollama server.
Update the chatbot_function in chatbot.py to be generic and handle different providers.
Update main.py to include the Ollama provider and start the conversation graph.
Here's the updated ollama_provider.py and main.py files:

# providers/ollama_provider.py

from.llm_provider import LLMProvider
from langchain.llms import Ollama

class OllamaProvider(LLMProvider):
    def initialize_model(self, model_name="llama3.1", temperature=0):
        return Ollama(base_url="http://localhost:11434", model=model_name, temperature=temperature)

    def generate_response(self, model, prompt):
        return model(prompt)

provider

# main.py

from.chatbot.chatbot import chatbot_function
from.providers.groq_provider import GroqProvider
from.providers.gemini_provider import GeminiProvider
from.providers.ollama_provider import OllamaProvider
from.tools.tool_integration import create_tool_node
from.graph.graph_builder import build_conversation_graph

def main():
    groq_api_key = "your_groq_api_key"
    gemini_api_key = "your_gemini_api_key"
    system_message = "You are a helpful assistant."
    human_message = "Explain the importance of low latency LLMs."

    providers = {
        "groq": GroqProvider(),
        "gemini": GeminiProvider(),
        "ollama": OllamaProvider()
    }

    for provider_name, provider in providers.items():
        if provider_name == "ollama":
            chatbot_func = lambda state: chatbot_function(provider, None, system_message, human_message)
        else:
            chatbot_func = lambda state: chatbot_function(provider, groq_api_key if provider_name == "groq" else gemini_api_key, system_message, human_message)

    tool_node = create_tool_node(wikipedia_tool)
        graph = build_conversation_graph(chatbot_func, tool_node)
        graph.start()

if __name__ == "__main__":
    main()

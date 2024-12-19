import os
from typing import List, Dict
import json
from exa_py import Exa
import ollama

# Initialize Exa client
exa = Exa(os.environ["EXA_API_KEY"])

# Define Exa search function as a tool
def exa_search(query: str, num_results: int = 3) -> List[Dict]:
    response = exa.search_and_contents(
        query,
        num_results=num_results,
        text={"max_characters": 500},
        use_autoprompt=True
    )
    return [{"title": r.title, "url": r.url, "content": r.text} for r in response.results]

# Define the tool in Ollama's expected format
tools = [
    {
        "name": "exa_search",
        "description": "Search the web for up-to-date information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 3}
            },
            "required": ["query"]
        }
    }
]

# Function to handle tool calls
def handle_tool_call(tool_call):
    if tool_call['name'] == 'exa_search':
        args = json.loads(tool_call['arguments'])
        return json.dumps(exa_search(args['query'], args.get('num_results', 3)))
    return json.dumps({"error": "Unknown tool"})

# Main interaction loop
def chat_with_exa_augmented_ollama():
    print("Chat with Exa-augmented Ollama (type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant augmented with the ability to search the web for up-to-date information using the Exa search engine. Use the exa_search tool when you need current information."
            },
            {"role": "user", "content": user_input}
        ]

        response = ollama.chat(model='llama2', messages=messages, tools=tools)

        if 'tool_calls' in response:
            for tool_call in response['tool_calls']:
                tool_response = handle_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "content": tool_response,
                    "name": tool_call['name']
                })
            response = ollama.chat(model='llama2', messages=messages)

        print("Ollama:", response['message']['content'])

# Run the chat
chat_with_exa_augmented_ollama()
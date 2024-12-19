## LangGraph Cheat Sheet

**LangGraph is a framework within LangChain for building stateful, multi-actor applications with LLMs.[**[1](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcTS-LTuEPHoVw9sz-u1ieY7PSVlZKytB2ETWCRE0akcJI2gyPKy62wdvBcMg_8u9kRw5HPAACwVSLEo6Gz2hpLFqvRDxBKE-5GWPwVJjkZzsx-UH7m8HnSGI4zWuy45Q3CCKkFywdlmuWwxkwD5VwI=)] It excels in creating complex workflows, especially agent-based and multi-agent systems, by representing tasks as directed graphs.

**Core Concepts:**

* **State:** **A shared object that persists throughout the workflow execution. Nodes can read and modify the state, enabling communication and data sharing between different parts of the graph.**
* **Nodes:** **Represent individual actions or steps in the workflow. Nodes can be:**

  * **Functions:** **Regular Python functions.**
  * **Tools:** **Functions decorated with** **@tool** **from LangChain.**
  * **LLMs:** **Language models for text generation or other AI tasks.**
  * **Agents:** **Instances of** **Agent** **classes, combining LLMs, tools, and instructions.**
  * **Special Nodes:** **START** **and** **END** **nodes define the workflow's beginning and end.[**[2](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcS2JEOFnSsRvUF85kfWjoR5Ln9Xzx3YXrMALYAzh8UbPv4_D68kAuJy9MEO5jCU-cw7qCL7JC0MFZ1xFStgMrZ9U6QiL_hxxgG8nN-JDx14WSIGBfWUm9uyK_RsTehaGwgNAZz8gViuLhtYDtndum-FxnXIS0rur5S8g-OEbrsS8xA0p5H8QF_bwpPxAyTa7zUS7Qc9d5P64q-dF3kQsaBc)]
* **Edges:** **Connections between nodes, defining the execution flow.[**[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQETLEdjiD0jG3ypEB0XO2QJERIF6thvmXTHMdzJABQ8qEhzAtmZTVU3jJNwR4NHzlCDT91P3ohKE0Bfj-iPd0ymO1udtYpqn-TpRGN5IZYS-GnXN_0VdaXiLZCAsnVY7D4dVAt5SGgBXej1jBxydNzZu9SELOI2jU4cvEF5AmswqlQGrFQCt_wrJdPf6d94bMP8w==)] Edges can be:

  * **Basic Edges:** **Simple connections from one node to another.[**[3](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcQETLEdjiD0jG3ypEB0XO2QJERIF6thvmXTHMdzJABQ8qEhzAtmZTVU3jJNwR4NHzlCDT91P3ohKE0Bfj-iPd0ymO1udtYpqn-TpRGN5IZYS-GnXN_0VdaXiLZCAsnVY7D4dVAt5SGgBXej1jBxydNzZu9SELOI2jU4cvEF5AmswqlQGrFQCt_wrJdPf6d94bMP8w==)]
  * **Conditional Edges:** **Control the flow based on conditions evaluated on the state.**

**Building a LangGraph Workflow:**

* **Define Nodes:** **Create functions, tools, or other callable objects representing the actions in your workflow.**
* **Create a** **StateGraph**: **Instantiate a** **StateGraph** **object to manage the workflow.**
* **Add Nodes to the Graph:** **Use** **add_node()** **to add the defined nodes to the graph, specifying a unique name for each node.**
* **Add Edges to the Graph:** **Use** **add_edge()** **to connect nodes. For basic edges, provide the source and destination node names. For conditional edges, provide a condition function that evaluates the state.**
* **(Optional) Set Entry Point:** **If the starting node is not** **START**, use **set_entry_point()** **to specify the initial node.**
* **Execute the Workflow:** **Call** **run()** **on the** **StateGraph** **instance, providing an initial state if necessary.**

**Code Example:**

```
from langgraph import StateGraph
from langchain.tools import DuckDuckGoSearchRun

# Define nodes (functions)
def get_search_query(state):
    state["query"] = input("Enter your search query: ")
    return state

def search_web(state):
    search = DuckDuckGoSearchRun()
    state["results"] = search.run(state["query"])
    return state

def process_results(state):
    print(state["results"])
    return state

# Create a StateGraph
workflow = StateGraph()

# Add nodes
workflow.add_node("get_query", get_search_query)
workflow.add_node("search", search_web)
workflow.add_node("process", process_results)

# Add edges
workflow.add_edge("START", "get_query")
workflow.add_edge("get_query", "search")
workflow.add_edge("search", "process")
workflow.add_edge("process", "END")

# Execute the workflow
workflow.run()
```

**Conditional Edges Example:**

```
from langgraph import StateGraph

# ... (previous node definitions)

def check_results(state):
    return len(state["results"]) > 0

# ... (add nodes as before)

# Add conditional edges
workflow.add_edge("search", "process", condition=check_results)
workflow.add_edge("search", "get_query", condition=lambda state: not check_results(state)) # Replan if no results

# ... (execute workflow)
```

**Key Considerations:**

* **State Management:** **Carefully design the state structure to effectively share data between nodes.**
* **Error Handling:** **Implement appropriate error handling within nodes and consider using conditional edges to manage errors gracefully.**
* **Cycles:** **LangGraph supports cycles in the graph, enabling complex iterative workflows.[**[4](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AYygrcTk1hK2xcJdUUd-S2lw1vm-1CkSmDeirb_kqGiYZcStLX_XKJsmysxVfLtnstCtocoqt8kKqnt6s_g1AGEBQV2OY1G3GJ4EervE56FJZCIbvOc2YPquJZ7y_2skG-gd4T3Q5PbgCps2JzR2tdjF7TFU-n4nVlO18Aknvj5yV52gpjWJSRNRc4jgPuXlv_OoP22HmSEpJInq5a-OSiGvlwRMEy8Lyvry2HQvFZj4Pq0NmW2HAEDsGJ-gCEXxApr4W05L0VV4EwcS-XZiNA==)] However, ensure that cycles terminate appropriately to avoid infinite loops.

**This cheat sheet provides a concise overview of LangGraph's core functionality. Refer to the official LangChain documentation for detailed information and advanced features. As of today, December 11, 2024, this information is current, but future updates to LangGraph may introduce changes.**

### Create a graph

```
graph = Graph()
```

### Visualize the graph (requires graphviz)

```
graph.draw("my_graph.png")
```

### Load and save graphs

```
graph.save("my_graph.json")
loaded_graph = Graph.load("my_graph.json")
```


Use code with caution.

Python

Example with LangChain Integration:

from langchain.chat_models import ChatOpenAI

from langchain.prompts import ChatPromptTemplate

from langchain.schema import StrOutputParser

from langgraph import Graph, Node

# Define LangChain components

prompt = ChatPromptTemplate.from_template("Translate '{text}' into French.")

llm = ChatOpenAI()

chain = prompt | llm | StrOutputParser()

# Create a LangGraph node using the LangChain chain

translation_node = Node(chain, name="Translate to French")

# Build and execute the graph (add other nodes and edges as needed)

graph = Graph()

graph.add_node(translation_node)

# ... (Add more nodes and edges) ...

inputs = {"Translate to French": {"text": "Hello, world!"}}

outputs = graph.execute(inputs)

print(outputs) # Output: {'Translate to French': {'text': 'Bonjour le monde!'}}  (Output may vary slightly)

Use code with caution.

Python

Advanced Usage:

Conditional Execution: Implement branching logic based on node outputs.

Error Handling: Define error handling mechanisms for individual nodes or the entire graph.

Asynchronous Execution: Execute nodes concurrently for improved performance. (Explore graph.aexecute() and related methods).

Custom Node Types: Extend langgraph.node.Node to create nodes with specialized behavior.

This cheat sheet provides a basic overview. Refer to the official LangGraph documentation for detailed information and advanced features. Keep in mind that LangGraph is under active development, so some APIs and functionalities might evolve.

from flask import Flask, request, jsonify, abort, render_template
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
import google.generativeai as genai
import os
import logging
import asyncio
from functools import lru_cache
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from json_storage import JSONStorage, agents, workflows, tools
import networkx as nx

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    genai.configure(api_key="AIzaSyAsHz1B6g-Ta5nxqszAu-wPahOP0x5Wfko")
    model = genai.GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(history=[])
except Exception as e:
    logger.error(f"Failed to initialize Gemini API: {str(e)}")
    raise


@lru_cache(maxsize=100)
def cached_chat_with_model(message):
    return chat_with_model(message)


def generate_text(prompt):
    response = model.generate_content(prompt)
    return response.text


def chat_with_model(message):
    response = chat.send_message(message)
    return response.text


def create_agent_node(agent_config):
    def agent_function(state):
        messages = state['messages']
        model_type = agent_config.get('model_type', 'gemini')
        model_name = agent_config.get('model', 'gemini-1.5-pro')

        try:
            if model_type == 'gemini':
                response = chat_with_model(messages[-1].content)
            elif model_type == 'ollama':
                response = f"Ollama model {model_name} response (not implemented)"
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            if 'post_processing' in agent_config:
                response = eval(agent_config['post_processing'])(response)

            logger.info(f"Agent using {model_type} model {model_name} responded successfully")
            return {"messages": messages + [HumanMessage(content=response)]}
        except Exception as e:
            logger.error(f"Error in agent function: {str(e)}")
            return {"messages": messages + [HumanMessage(content="Error occurred in agent processing")]}

    return agent_function


def to_uppercase(text):
    return text.upper()


def add_prefix(text, prefix="AI: "):
    return prefix + text


import networkx as nx


def create_workflow_graph(workflow_config):
    graph = StateGraph(MessagesState)
    dependency_graph = nx.DiGraph()

    for agent in workflow_config['agents']:
        graph.add_node(agent['name'], create_agent_node(agent))
        dependency_graph.add_node(agent['name'])

    start_nodes = []
    for edge in workflow_config['edges']:
        from_node = edge['from']
        to_node = edge['to']

        if from_node == 'start':
            start_nodes.append(to_node)
        elif to_node == 'end':
            graph.add_edge(from_node, END)
        else:
            graph.add_edge(from_node, to_node)
            dependency_graph.add_edge(from_node, to_node)

    independent_agents = [node for node in dependency_graph.nodes() if dependency_graph.in_degree(node) == 0]

    if len(independent_agents) > 1:
        async def parallel_execution(state):
            tasks = [create_agent_node(next(agent for agent in workflow_config['agents'] if agent['name'] == name))(state) for name in independent_agents]
            results = await asyncio.gather(*tasks)
            combined_messages = state['messages']
            for result in results:
                combined_messages.extend(result['messages'])
            return {"messages": combined_messages}

        graph.add_node("parallel_execution", parallel_execution)
        for agent in independent_agents:
            graph.add_edge("parallel_execution", agent)
        graph.set_entry_point("parallel_execution")
    elif start_nodes:
        graph.set_entry_point(start_nodes[0])

    return graph.compile()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api')
def welcome():
    return "Welcome to the LLM Agents and Workflows API"


@app.route('/tools', methods=['GET', 'POST'])
def manage_tools():
    if request.method == 'GET':
        return jsonify(tools.read_all()), 200

    elif request.method == 'POST':
        new_tool = request.json
        created_tool = tools.create(new_tool)
        return jsonify({"message": "Tool created successfully", "tool": created_tool}), 201


@app.route('/tools/<int:tool_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_tool(tool_id):
    if request.method == 'GET':
        tool = tools.read(tool_id)
        if tool:
            return jsonify(tool), 200
        return jsonify({"message": "Tool not found"}), 404

    elif request.method == 'PUT':
        updated_tool = request.json
        result = tools.update(tool_id, updated_tool)
        if result:
            return jsonify({"message": "Tool updated successfully", "tool": result}), 200
        return jsonify({"message": "Tool not found"}), 404

    elif request.method == 'DELETE':
        tools.delete(tool_id)
        return jsonify({"message": "Tool deleted successfully"}), 200


@app.route('/agents', methods=['GET', 'POST'])
def manage_agents():
    if request.method == 'GET':
        return jsonify(agents.read_all()), 200

    elif request.method == 'POST':
        new_agent = request.json
        created_agent = agents.create(new_agent)
        return jsonify({"message": "Agent created successfully", "agent": created_agent}), 201


@app.route('/agents/<int:agent_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_agent(agent_id):
    if request.method == 'GET':
        agent = agents.read(agent_id)
        if agent:
            return jsonify(agent), 200
        return jsonify({"message": "Agent not found"}), 404

    elif request.method == 'PUT':
        updated_agent = request.json
        result = agents.update(agent_id, updated_agent)
        if result:
            return jsonify({"message": "Agent updated successfully", "agent": result}), 200
        return jsonify({"message": "Agent not found"}), 404

    elif request.method == 'DELETE':
        agents.delete(agent_id)
        return jsonify({"message": "Agent deleted successfully"}), 200


@app.route('/workflows', methods=['GET', 'POST'])
def manage_workflows():
    if request.method == 'GET':
        return jsonify(workflows.read_all()), 200

    elif request.method == 'POST':
        new_workflow = request.json
        created_workflow = workflows.create(new_workflow)
        return jsonify({"message": "Workflow created successfully", "workflow": created_workflow}), 201


@app.route('/workflows/<int:workflow_id>', methods=['GET', 'PUT', 'DELETE'])
def manage_workflow(workflow_id):
    if request.method == 'GET':
        workflow = workflows.read(workflow_id)
        if workflow:
            return jsonify(workflow), 200
        return jsonify({"message": "Workflow not found"}), 404

    elif request.method == 'PUT':
        updated_workflow = request.json
        result = workflows.update(workflow_id, updated_workflow)
        if result:
            return jsonify({"message": "Workflow updated successfully", "workflow": result}), 200
        return jsonify({"message": "Workflow not found"}), 404

    elif request.method == 'DELETE':
        workflows.delete(workflow_id)
        return jsonify({"message": "Workflow deleted successfully"}), 200


@app.route('/execute_workflow/<int:workflow_id>', methods=['POST'])
async def execute_workflow(workflow_id):
    try:
        workflow = workflows.read(workflow_id)
        if not workflow:
            logger.warning(f"Workflow with id {workflow_id} not found")
            return jsonify({"message": "Workflow not found"}), 404

        user_input = request.json.get('input', '')
        logger.info(f"Executing workflow {workflow_id} with input: {user_input}")

        graph = create_workflow_graph(workflow)
        result = await graph.ainvoke({"messages": [HumanMessage(content=user_input)]})

        logger.info(f"Workflow {workflow_id} executed successfully")
        return jsonify({"result": result['messages'][-1].content}), 200
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
        return jsonify({"message": "An error occurred while executing the workflow"}), 500


if __name__ == '__main__':
    app.run(debug=True)
Use code with caution.
Python
import json
import os

class JSONStorage:
    def __init__(self, file_name):
        self.file_name = file_name
        if not os.path.exists(file_name):
            with open(file_name, 'w') as f:
                json.dump([], f)

    def read_all(self):
        with open(self.file_name, 'r') as f:
            return json.load(f)

    def write_all(self, data):
        with open(self.file_name, 'w') as f:
            json.dump(data, f, indent=2)

    def create(self, item):
        data = self.read_all()
        if not data:
            item['id'] = 1
        else:
            item['id'] = max(d['id'] for d in data) + 1
        data.append(item)
        self.write_all(data)
        return item

    def read(self, id):
        data = self.read_all()
        return next((item for item in data if item['id'] == id), None)

    def update(self, id, updated_item):
        data = self.read_all()
        for item in data:
            if item['id'] == id:
                item.update(updated_item)
                self.write_all(data)
                return item
        return None

    def delete(self, id):
        data = self.read_all()
        data = [item for item in data if item['id'] != id]
        self.write_all(data)

agents = JSONStorage('agents.json')
workflows = JSONStorage('workflows.json')
tools = JSONStorage('tools.json')
Use code with caution.
Python
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Agents and Workflows</title>
    <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        h1, h2 { color: #333; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="text"], textarea { width: 100%; padding: 8px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #workflowList, #result { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>LLM Agents and Workflows</h1>
        
        <h2>Create Workflow</h2>
        <div class="form-group">
            <label for="workflowName">Workflow Name:</label>
            <input type="text" id="workflowName" required>
        </div>
        <div class="form-group">
            <label for="workflowConfig">Workflow Configuration (JSON):</label>
            <textarea id="workflowConfig" rows="10" required></textarea>
        </div>
        <button onclick="createWorkflow()">Create Workflow</button>
        
        <h2>Execute Workflow</h2>
        <div class="form-group">
            <label for="workflowId">Workflow ID:</label>
            <input type="text" id="workflowId" required>
        </div>
        <div class="form-group">
            <label for="workflowInput">Input:</label>
            <input type="text" id="workflowInput" required>
        </div>
        <button onclick="executeWorkflow()">Execute Workflow</button>
        
        <div id="result"></div>
        
        <h2>Workflows</h2>
        <button onclick="listWorkflows()">Refresh Workflow List</button>
        <div id="workflowList"></div>
    </div>

    <script>
        function createWorkflow() {
            const name = document.getElementById('workflowName').value;
            const config = JSON.parse(document.getElementById('workflowConfig').value);
            axios.post('/workflows', { name, ...config })
                .then(response => {
                    alert('Workflow created successfully');
                    listWorkflows();
                })
                .catch(error => {
                    alert('Error creating workflow: ' + error.response.data.message);
                });
        }

        function executeWorkflow() {
            const workflowId = document.getElementById('workflowId').value;
            const input = document.getElementById('workflowInput').value;
            axios.post(`/execute_workflow/${workflowId}`, { input })
                .then(response => {
                    document.getElementById('result').innerHTML = '<h3>Result:</h3><pre>' + JSON.stringify(response.data, null, 2) + '</pre>';
                })
                .catch(error => {
                    alert('Error executing workflow: ' + error.response.data.message);
                });
        }

        function listWorkflows() {
            axios.get('/workflows')
                .then(response => {
                    const workflowList = document.getElementById('workflowList');
                    workflowList.innerHTML = '<h3>Available Workflows:</h3>';
                    response.data.forEach(workflow => {
                        workflowList.innerHTML += `<p>ID: ${workflow.id}, Name: ${workflow.name}</p>`;
                    });
                })
                .catch(error => {
                    alert('Error fetching workflows: ' + error.response.data.message);
                });
        }

        listWorkflows();
    </script>
</body>
</html>
Use code with caution.
Html
import pytest
import json
from app import app
from json_storage import JSONStorage, agents, workflows, tools
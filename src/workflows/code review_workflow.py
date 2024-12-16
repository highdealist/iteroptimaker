import json
import os
import time
from click import prompt
import google.generativeai as genai
import re
from agents import ModelFactory # Import ModelFactory
from typing import List, Dict, Any
from agents.agent import BaseAgent as Agent
from models.model_manager import ModelManager
from tools.tool_manager import ToolManager
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path: List[str], mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  """Waits for the given files to be active.

  Some files uploaded to the Gemini API need to be processed before they can be
  used as prompt inputs. The status can be seen by querying the file's "state"
  field.

  This implementation uses a simple blocking polling loop. Production code
  should probably employ a more sophisticated approach.
  """
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# Create the model
generation_config = {
  "temperature": 0.3,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

with open("agents.json", "r", encoding="utf-8", errors="ignore") as f:
  agents = json.load(f)
  print(agents["researcher"])
  researcher_config = agents["researcher"]

coder_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="""Conduct a thorough analysis to identify vulnerabilities, glitches, and performance issues, focusing on memory management, resource allocation, and concurrency to ensure robustness and efficiency.

Diagnostic Reports:

Provide detailed reports for each issue, describing the problem, its impact, and the relevant code snippets.
Highlight patterns or recurring issues indicating broader systemic problems.
Actionable Recommendations:

Offer actionable solutions with best practices to prevent similar future issues.
Steps:

Comprehensive Code Review:

Analyze the entire codebase to detect errors, inefficiencies, and logical inconsistencies.
Scrutinize algorithms, data structures, and control flow using advanced techniques to uncover vulnerabilities and performance bottlenecks.
Diagnostic Reports:

Identify and explain logical errors, structural weaknesses, inefficiencies, security vulnerabilities, and performance bottlenecks.
Include code snippets showing the source of each issue.
Actionable Recommendations:

Provide specific, actionable solutions aligned with best practices and ensure feasibility within the current codebase.
Revised Source Code Snippets:

Offer optimized, functional code revisions with comments for clarity.
Clearly mark what to remove, replace, or retain, avoiding vague comments like “previous code unchanged.”
Recap the changes and their impact, noting any unintended side effects for further revision.
Deliverables:

A report detailing findings.
Actionable recommendations.
Revised code snippets with annotations.
A summary of changes and their impact.""") 

researcher_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="agents.researcher"
)
print(researcher_model._system_instruction)
model = ModelFactory.create_model(researcher_config["instruction"], **researcher_config)

# TODO Make these files available on the local file system
# You may need to update the file paths
files = [
  upload_to_gemini("gui.py", mime_type="text/x-python"),
  upload_to_gemini("models.py", mime_type="text/x-python"),
  upload_to_gemini("search_manager.py", mime_type="text/x-python"),
  upload_to_gemini("main.py", mime_type="text/x-python"),
  upload_to_gemini("tools.py", mime_type="text/x-python"),
  upload_to_gemini("agents.py", mime_type="text/x-python"),
  upload_to_gemini("llm_providers.py", mime_type="text/x-python"),
  upload_to_gemini("config.py", mime_type="text/x-python"),
]

# Some files have a processing delay. Wait for them to be ready.
wait_for_files_active(files)

chat_session = model.start_chat(
  history=[
    {
      "role": "user",
      "parts": [
        files[0],
        files[1],
        files[2],
        files[3],
        files[4],
        files[5],
        files[6],
        files[7],
        "This code aims to create a creative AI writing assistant that leverages the power of LLMs and a sophisticated agent-based architecture. It distinguishes itself through:\n\n- **Specialized Agents:** You've implemented a `researcher_agent` and a `writer_agent`, each with distinct roles and capabilities. This division of labor allows for more focused and effective task execution.\n- **Contextual Tool Use:** Your agents dynamically determine and utilize the most appropriate tools from the `ToolManager` based on the user's input and the ongoing context. This adaptability is key to a versatile and powerful assistant.\n- **Flexible Model Selection:** The `ModelManager` allows you to easily switch between different LLM providers (OpenAI, Gemini) and models, providing flexibility for experimentation and optimization.\n- **Web Search Integration:** The `SearchManager` enables access to real-world information through web searches, enriching the assistant's knowledge base.\n- **Interactive GUI:** The Tkinter-based GUI offers a user-friendly interface for interacting with the assistant.\n\n**Functionality:**\n\n1. **User Interaction:** Users input prompts or instructions through the GUI.\n2. **Agent Selection:** The application routes the user's request to the most suitable agent (researcher or writer) based on the task at hand.\n3. **Dynamic Tool Invocation:**  Agents analyze the user's input and the current context to determine which tools, if any, are required, and execute them accordingly.\n4. **LLM-Powered Response Generation:** The selected agent interacts with the chosen LLM, incorporating information gathered from tools and previous interactions to generate a comprehensive and contextually relevant response.\n5. **Output Display:** The generated text, along with any relevant findings from tool executions, is presented to the user in the GUI.\n\n**",
      ],
    },
    {
      "role": "model",
      "parts": [
          "I see you have uploaded several python files, what would you like me to do?"
        ],
    },
  ]
)

prompt = "Please thoroughly inspect the code and identify all redundancies, unnecessary abstractions, logical inconsistencies, and other flaws or bugs and list them in a numbered list in the format:   1. detailed description of first problem 2. detailed description of second problem 3. etc... so that your answer can be properly parsed."

#Function to parse the listed problems from the response to the prompt

def parse_listed_problems(response):
  return [
      # Split each line into a problem number and a description, and strip any whitespace
      # from the description.  We use a regular expression to match lines that start
      # with a number followed by a period and a space, which should be the format
      # of the listed problems.
      re.sub(r"^\d+\.\s", "", line).strip()
      for line in response.text.split("\n")
      if re.match(r"^\d+\.\s", line)  # Only match lines that start with a number followed by a period        
  ]

    
#"Please thoroughly inspect the code and determine which method of storing the system_instructions / system prompts for each of the agents.  Should I use json?  Python dictionary?  Which method is the best for being able to create and modify agents on the fly through the GUI when the program is running?"

class CodeReviewWorkflow:
    """
    Implements a comprehensive code review workflow with multiple specialized agents
    simulating different roles in a software development team.
    """
    def __init__(self, model_manager: ModelManager, tool_manager: ToolManager):
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        
        # Initialize specialized agents for different roles
        self.developer_agent = Agent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="developer",
            instruction="""You are a skilled software developer responsible for implementing features and fixing bugs.
            Your tasks include:
            1. Writing clean, efficient, and well-documented code
            2. Addressing review feedback and implementing fixes
            3. Explaining implementation decisions and trade-offs
            4. Ensuring code meets project standards and requirements""",
            tools=["python_repl", "file_operations"],
            model_config={"temperature": 0.3},
            name="Developer"
        )
        
        self.code_reviewer_agent = Agent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="code_reviewer",
            instruction="""You are an experienced code reviewer focused on code quality and best practices.
            Your responsibilities include:
            1. Reviewing code for clarity, efficiency, and maintainability
            2. Identifying potential bugs and edge cases
            3. Suggesting improvements in code structure and organization
            4. Ensuring adherence to coding standards and patterns""",
            tools=["static_analysis", "code_quality_metrics"],
            model_config={"temperature": 0.2},
            name="Code Reviewer"
        )
        
        self.security_reviewer_agent = Agent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="security_reviewer",
            instruction="""You are a security expert focused on identifying and preventing security vulnerabilities.
            Your tasks include:
            1. Conducting security analysis of code changes
            2. Identifying potential security risks and vulnerabilities
            3. Recommending security best practices
            4. Ensuring compliance with security standards""",
            tools=["security_scanner", "vulnerability_analysis"],
            model_config={"temperature": 0.1},
            name="Security Reviewer"
        )
        
        self.tech_lead_agent = Agent(
            model_manager=model_manager,
            tool_manager=tool_manager,
            agent_type="tech_lead",
            instruction="""You are a technical lead responsible for overall code quality and architecture.
            Your responsibilities include:
            1. Reviewing architectural decisions and their implications
            2. Ensuring technical consistency across the codebase
            3. Making final decisions on technical disputes
            4. Approving or requesting changes to pull requests""",
            tools=["architecture_analysis", "dependency_checker"],
            model_config={"temperature": 0.2},
            name="Tech Lead"
        )

    def review_code(self, code_files: List[str]) -> Dict[str, Any]:
        """
        Orchestrates the code review process across all agents.
        
        Args:
            code_files: List of files to review
            
        Returns:
            Dictionary containing review results from all agents
        """
        review_results = {
            "developer_notes": [],
            "code_review": [],
            "security_review": [],
            "tech_lead_review": [],
            "final_decision": None
        }
        
        # Step 1: Developer self-review
        developer_response = self.developer_agent.analyze(
            {"task": "self_review", "files": code_files}
        )
        review_results["developer_notes"] = developer_response
        
        # Step 2: Parallel reviews by code and security reviewers
        code_review = self.code_reviewer_agent.analyze(
            {"task": "code_review", "files": code_files}
        )
        review_results["code_review"] = code_review
        
        security_review = self.security_reviewer_agent.analyze(
            {"task": "security_review", "files": code_files}
        )
        review_results["security_review"] = security_review
        
        # Step 3: Tech Lead final review and decision
        tech_lead_review = self.tech_lead_agent.analyze({
            "task": "final_review",
            "files": code_files,
            "code_review": code_review,
            "security_review": security_review
        })
        review_results["tech_lead_review"] = tech_lead_review
        
        # Determine final decision
        review_results["final_decision"] = tech_lead_review.get("decision", "needs_revision")
        
        return review_results

    def handle_review_feedback(self, review_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes review feedback and coordinates necessary changes.
        
        Args:
            review_results: Results from the review process
            
        Returns:
            Dictionary containing response to feedback and any changes made
        """
        feedback_response = {
            "changes_made": [],
            "pending_items": [],
            "status": "in_progress"
        }
        
        # Developer addresses feedback
        if review_results["final_decision"] == "needs_revision":
            developer_changes = self.developer_agent.analyze({
                "task": "address_feedback",
                "review_results": review_results
            })
            feedback_response["changes_made"] = developer_changes.get("changes", [])
            feedback_response["pending_items"] = developer_changes.get("pending", [])
            
        feedback_response["status"] = "completed" if not feedback_response["pending_items"] else "in_progress"
        return feedback_response

def create_code_review_workflow(model_manager: ModelManager, tool_manager: ToolManager) -> CodeReviewWorkflow:
    """
    Factory function to create a new code review workflow instance.
    """
    return CodeReviewWorkflow(model_manager, tool_manager)

def code_review_workflow():
  while True:
      response = chat_session.send_message(prompt)
      print(response.text)
      problems = parse_listed_problems(response)

    #Iteratively send each problem to the agent and print the response

      for problem in problems:
        print(f"Problem: {problem}")
        response = chat_session.send_message(problem)
        print(f"Response: {response}")

#Create a simple CLI command menu for running different functions / workflows

def main():
    model_manager = ModelManager()
    tool_manager = ToolManager()
    workflow = create_code_review_workflow(model_manager, tool_manager)
    
    while True:
        print("\nCode Review Workflow Menu:")
        print("1. Start New Code Review")
        print("2. Check Review Status")
        print("3. Process Review Feedback")
        print("4. Exit")
        
        choice = input("Enter your choice: ")
        
        if choice == "1":
            files = input("Enter comma-separated list of files to review: ").split(",")
            files = [f.strip() for f in files]
            results = workflow.review_code(files)
            print("\nReview Results:")
            print(json.dumps(results, indent=2))
            
        elif choice == "2":
            # Add status checking functionality here
            print("Status checking not implemented yet")
            
        elif choice == "3":
            # Add feedback processing functionality here
            print("Feedback processing not implemented yet")
            
        elif choice == "4":
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

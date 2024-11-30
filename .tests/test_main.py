import unittest
import logging
import tkinter as tk
from unittest.mock import MagicMock, patch
from tkinter import messagebox
from main import AIAssistantApp, main
from config import (
    GEMINI_API_KEY,
    MAX_SEARCH_RESULTS,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    GEMINI_PRO_MODEL
)
from models import ModelManager
from agents import AgentManager, ToolManager
from search_manager import initialize_search_manager
from gui.app import App
from llm_providers import OpenAIProvider, GeminiProvider
import tools
import json
from functools import partial
from langgraph.graph import END  # Import END from langgraph

class TestAIAssistantApp(unittest.TestCase):

    @patch('main.initialize_search_manager')
    @patch('main.ToolManager')
    @patch('main.ModelManager')
    @patch('main.AgentManager')
    @patch('main.OpenAIProvider')
    @patch('main.GeminiProvider')
    @patch('main.App')
    @patch('main.json.load')
    @patch('main.messagebox.showerror')
    def test_init(self, mock_showerror, mock_json_load, mock_app, mock_gemini_provider, mock_openai_provider,
                 mock_agent_manager, mock_model_manager, mock_tool_manager, mock_initialize_search_manager):
        """Tests that AIAssistantApp initializes without errors."""
        mock_json_load.return_value = {"researcher": {"agent_type": "producer", "system_prompt": "test instruction"},
                                        "writer": {"agent_type": "utility", "system_prompt": "test instruction"}}
        mock_initialize_search_manager.return_value = MagicMock()
        mock_tool_manager.return_value = MagicMock()
        mock_model_manager.return_value = MagicMock()
        mock_agent_manager.return_value = MagicMock()
        mock_openai_provider.return_value = MagicMock()
        mock_gemini_provider.return_value = MagicMock()
        mock_app.return_value = MagicMock()

        app = AIAssistantApp()
        self.assertIsNotNone(app)
        mock_initialize_search_manager.assert_called_once()
        mock_tool_manager.assert_called_once()
        mock_model_manager.assert_called_once()
        mock_agent_manager.assert_called_once()
        mock_openai_provider.assert_called_once()
        mock_gemini_provider.assert_called_once()
        mock_app.assert_called_once()
        mock_json_load.assert_called_once()
        mock_showerror.assert_not_called()

    @patch('main.messagebox.showerror')
    def test_run(self, mock_showerror):
        """Tests that AIAssistantApp.run starts the main loop without errors."""
        app = AIAssistantApp()
        app.app = MagicMock()
        app.app.mainloop = MagicMock()
        app.run()
        app.app.mainloop.assert_called_once()
        mock_showerror.assert_not_called()

    @patch('main.messagebox.showerror')
    def test_run_with_error(self, mock_showerror):
        """Tests that AIAssistantApp.run handles errors in the main loop."""
        app = AIAssistantApp()
        app.app = MagicMock()
        app.app.mainloop.side_effect = Exception("Test error")
        app.run()
        mock_showerror.assert_called_once_with("Error", "An unexpected error occurred: Test error")

    @patch('main.messagebox.showerror')
    def test_main(self, mock_showerror):
        """Tests that main function starts the application without errors."""
        with patch('main.AIAssistantApp') as mock_app:
            mock_app.return_value = MagicMock()
            mock_app.return_value.run = MagicMock()
            main()
            mock_app.assert_called_once()
            mock_app.return_value.run.assert_called_once()
            mock_showerror.assert_not_called()

    @patch('main.messagebox.showerror')
    def test_main_with_error(self, mock_showerror):
        """Tests that main function handles errors during application startup."""
        with patch('main.AIAssistantApp') as mock_app:
            mock_app.side_effect = Exception("Test error")
            main()
            mock_showerror.assert_called_once_with("Fatal Error", "An unexpected error occurred: Test error")

class TestApp(unittest.TestCase):

    @patch('gui.app.initialize_search_manager')
    @patch('gui.app.ToolManager')
    @patch('gui.app.ModelManager')
    @patch('gui.app.AgentManager')
    @patch('gui.app.OpenAIProvider')
    @patch('gui.app.GeminiProvider')
    def test_init(self, mock_gemini_provider, mock_openai_provider, mock_agent_manager, mock_model_manager,
                 mock_tool_manager, mock_initialize_search_manager):
        """Tests that App initializes without errors."""
        mock_initialize_search_manager.return_value = MagicMock()
        mock_tool_manager.return_value = MagicMock()
        mock_model_manager.return_value = MagicMock()
        mock_agent_manager.return_value = MagicMock()
        mock_openai_provider.return_value = MagicMock()
        mock_gemini_provider.return_value = MagicMock()

        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        self.assertIsNotNone(app)
        mock_initialize_search_manager.assert_called_once()
        mock_tool_manager.assert_called_once()
        mock_model_manager.assert_called_once()
        mock_agent_manager.assert_called_once()
        mock_openai_provider.assert_called_once()
        mock_gemini_provider.assert_called_once()

    @patch('gui.app.messagebox.showerror')
    def test_send_message_with_invalid_agent(self, mock_showerror):
        """Tests that send_message handles invalid agent selection."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.agent_var.set("invalid_agent")
        app.input_area.get = MagicMock(return_value="Test input")
        app.append_chat = MagicMock()
        app.process_user_input = MagicMock()
        app.send_message()
        mock_showerror.assert_called_once_with("Error", "Invalid agent selected.")
        app.append_chat.assert_not_called()
        app.process_user_input.assert_not_called()

    @patch('gui.app.messagebox.showerror')
    def test_process_user_input_with_web_search(self, mock_showerror):
        """Tests that process_user_input handles web search requests."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.search_manager.search = MagicMock(return_value="Test search results")
        app.append_chat = MagicMock()
        app.model_manager.generate_response = MagicMock(return_value="WEB_SEARCH: test query")
        app.researcher_agent.tools = ["web_search"]
        response = app.process_user_input("Test input", app.researcher_agent)
        self.assertEqual(response, "Test search results")
        app.search_manager.search.assert_called_once_with("test query", num_results=10)
        app.append_chat.assert_not_called()
        mock_showerror.assert_not_called()

    @patch('gui.app.messagebox.showerror')
    def test_process_user_input_with_web_search_tool_not_available(self, mock_showerror):
        """Tests that process_user_input handles web search requests when the tool is not available."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.search_manager.search = MagicMock()
        app.append_chat = MagicMock()
        app.model_manager.generate_response = MagicMock(return_value="WEB_SEARCH: test query")
        app.researcher_agent.tools = []
        response = app.process_user_input("Test input", app.researcher_agent)
        self.assertEqual(response, "Error: Web search tool not available.")
        app.search_manager.search.assert_not_called()
        app.append_chat.assert_not_called()
        mock_showerror.assert_called_once_with("Error", "Web search tool not available.")

    @patch('gui.app.messagebox.showerror')
    def test_process_user_input_with_web_search_no_search_manager(self, mock_showerror):
        """Tests that process_user_input handles web search requests when there is no search manager."""
        app = App(search_manager=None, tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.append_chat = MagicMock()
        app.model_manager.generate_response = MagicMock(return_value="WEB_SEARCH: test query")
        app.researcher_agent.tools = ["web_search"]
        response = app.process_user_input("Test input", app.researcher_agent)
        self.assertEqual(response, "Error: Web search tool not available.")
        app.append_chat.assert_not_called()
        mock_showerror.assert_called_once_with("Error", "Web search tool not available.")

    @patch('gui.app.filedialog.askopenfilename')
    def test_open_chat(self, mock_askopenfilename):
        """Tests that open_chat opens a chat file."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.chat_area.config = MagicMock()
        app.chat_area.delete = MagicMock()
        app.chat_area.insert = MagicMock()
        mock_askopenfilename.return_value = "test.txt"
        with patch('gui.app.open') as mock_open:
            mock_open.return_value = MagicMock()
            app.open_chat()
            mock_askopenfilename.assert_called_once()
            mock_open.assert_called_once_with("test.txt", "r", encoding="utf-8")
            app.chat_area.config.assert_called_once_with(state=tk.NORMAL)
            app.chat_area.delete.assert_called_once_with("1.0", tk.END)
            app.chat_area.insert.assert_called_once()
            app.chat_area.config.assert_called_with(state=tk.DISABLED)

    @patch('gui.app.filedialog.asksaveasfilename')
    def test_save_chat(self, mock_asksaveasfilename):
        """Tests that save_chat saves the chat to a file."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.chat_area.get = MagicMock(return_value="Test chat content")
        mock_asksaveasfilename.return_value = "test.txt"
        with patch('gui.app.open') as mock_open:
            mock_open.return_value = MagicMock()
            app.save_chat()
            mock_asksaveasfilename.assert_called_once()
            mock_open.assert_called_once_with("test.txt", "w", encoding="utf-8")
            app.chat_area.get.assert_called_once_with("1.0", tk.END)
            mock_open.return_value.write.assert_called_once_with("Test chat content")

    @patch('gui.app.StateGraph')
    @patch('gui.app.create_workflow')
    def test_run_workflow(self, mock_create_workflow, mock_state_graph):
        """Tests that run_workflow starts the workflow execution."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.input_area.get = MagicMock(return_value="Test input")
        app.append_chat = MagicMock()
        app.execute_workflow = MagicMock()
        mock_create_workflow.return_value = MagicMock()
        mock_state_graph.return_value = MagicMock()
        app.run_workflow()
        app.input_area.get.assert_called_once()
        app.append_chat.assert_called_once_with("You: Test input")
        app.execute_workflow.assert_called_once_with("Test input")
        mock_create_workflow.assert_called_once()
        mock_state_graph.assert_called_once()

    @patch('gui.app.StateGraph')
    @patch('gui.app.create_workflow')
    def test_execute_workflow_with_invalid_agent(self, mock_create_workflow, mock_state_graph):
        """Tests that execute_workflow handles invalid agent selection."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.agent_var.set("invalid_agent")
        app.append_chat = MagicMock()
        app.current_agent_label.config = MagicMock()
        mock_create_workflow.return_value = MagicMock()
        mock_state_graph.return_value = MagicMock()
        app.execute_workflow("Test input")
        app.append_chat.assert_not_called()
        app.current_agent_label.config.assert_not_called()
        mock_create_workflow.assert_called_once()
        mock_state_graph.assert_called_once()

    @patch('gui.app.StateGraph')
    @patch('gui.app.create_workflow')
    def test_execute_workflow(self, mock_create_workflow, mock_state_graph):
        """Tests that execute_workflow runs the workflow steps."""
        app = App(search_manager=MagicMock(), tool_manager=MagicMock(), model_manager=MagicMock(),
                  agent_manager=MagicMock(), llm_provider=MagicMock(), researcher_agent=MagicMock(),
                  writer_agent=MagicMock())
        app.append_chat = MagicMock()
        app.current_agent_label.config = MagicMock()
        mock_create_workflow.return_value = MagicMock()
        mock_state_graph.return_value = MagicMock(stream=MagicMock(return_value=[
            {"sender": "researcher", "messages": [{"content": "Test response 1"}]},
            {"sender": "writer", "messages": [{"content": "Test response 2"}]},
            {"sender": END, "messages": []}
        ]))
        app.execute_workflow("Test input")
        app.append_chat.assert_has_calls([
            unittest.mock.call("researcher: Test response 1"),
            unittest.mock.call("writer: Test response 2")
        ])
        app.current_agent_label.config.assert_has_calls([
            unittest.mock.call(text="Current Agent: researcher"),
            unittest.mock.call(text="Current Agent: writer"),
            unittest.mock.call(text="Current Agent: None")
        ])
        mock_create_workflow.assert_called_once()
        mock_state_graph.assert_called_once()

if __name__ == '__main__':
    unittest.main()

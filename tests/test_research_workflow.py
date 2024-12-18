import unittest
from unittest.mock import patch, MagicMock
from src.research_workflow import extract_tool_call, process_user_input
import logging
from io import StringIO

class TestResearchWorkflow(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)
        self.log_capture_string = StringIO()
        self.handler = logging.StreamHandler(self.log_capture_string)
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.log_capture_string.close()

    def get_log_messages(self):
        return self.log_capture_string.getvalue()

    def test_extract_tool_call_valid(self):
        text = "Tool: Web Search, Query: | What is the capital of France? |"
        expected = {"tool": "Web Search", "query": "What is the capital of France?"}
        self.assertEqual(extract_tool_call(text), expected)
        self.assertIn("Successfully extracted tool call", self.get_log_messages())

    def test_extract_tool_call_no_match(self):
        text = "This is a regular text without a tool call."
        self.assertIsNone(extract_tool_call(text))
        self.assertIn("No tool call found in text", self.get_log_messages())

    def test_extract_tool_call_extra_spaces(self):
        text = "Tool:  Web Search  ,  Query:  |  What is the capital of France?  |  "
        expected = {"tool": "Web Search", "query": "What is the capital of France?"}
        self.assertEqual(extract_tool_call(text), expected)
        self.assertIn("Successfully extracted tool call", self.get_log_messages())

    def test_extract_tool_call_invalid_format(self):
        text = "Tool:Web Search Query: | What is the capital of France? |"
        self.assertIsNone(extract_tool_call(text))
        self.assertIn("No tool call found in text", self.get_log_messages())

    @patch('src.research_workflow.initialize_search_manager')
    @patch('src.research_workflow.GeminiModel')
    def test_process_user_input_no_tool_call(self, mock_gemini_model, mock_search_manager):
        mock_search_manager.return_value.search.return_value = []
        mock_model_instance = MagicMock()
        mock_model_instance.chat.return_value = "This is a test response without a tool call."
        mock_gemini_model.return_value = mock_model_instance
        
        user_input = "What is the meaning of life?"
        response = process_user_input(user_input)
        self.assertEqual(response, "This is a test response without a tool call.")
        mock_gemini_model.assert_called_once()
        mock_search_manager.assert_called_once()
        mock_model_instance.chat.assert_called()
        self.assertIn("No tool call detected, returning researcher response", self.get_log_messages())

    @patch('src.research_workflow.initialize_search_manager')
    @patch('src.research_workflow.GeminiModel')
    def test_process_user_input_with_tool_call(self, mock_gemini_model, mock_search_manager):
        mock_search_manager.return_value.search.return_value = [{"title": "Test Title", "url": "test.com", "snippet": "Test Snippet", "content": "Test Content"}]
        mock_model_instance = MagicMock()
        mock_model_instance.chat.side_effect = [
            "Tool: Web Search, Query: | Test Query |",
            "This is a final response."
        ]
        mock_gemini_model.return_value = mock_model_instance
        
        user_input = "I need to search for something."
        response = process_user_input(user_input)
        self.assertEqual(response, "This is a final response.")
        mock_gemini_model.assert_called()
        mock_search_manager.assert_called_once()
        mock_model_instance.chat.assert_called()
        self.assertIn("Tool call detected", self.get_log_messages())
        self.assertIn("Executing web search with query", self.get_log_messages())

    @patch('src.research_workflow.initialize_search_manager')
    @patch('src.research_workflow.GeminiModel')
    def test_process_user_input_max_searches(self, mock_gemini_model, mock_search_manager):
        mock_search_manager.return_value.search.return_value = [{"title": "Test Title", "url": "test.com", "snippet": "Test Snippet", "content": "Test Content"}]
        mock_model_instance = MagicMock()
        mock_model_instance.chat.side_effect = [
            "Tool: Web Search, Query: | Query 1 |",
            "Tool: Web Search, Query: | Query 2 |",
            "Tool: Web Search, Query: | Query 3 |",
            "Final response after max searches."
        ]
        mock_gemini_model.return_value = mock_model_instance
        
        user_input = "Test user input."
        response = process_user_input(user_input)
        self.assertEqual(response, "Final response after max searches.")
        mock_gemini_model.assert_called()
        mock_search_manager.assert_called()
        self.assertEqual(mock_search_manager.return_value.search.call_count, 3)
        mock_model_instance.chat.assert_called()
        self.assertIn("Maximum searches reached, generating final response", self.get_log_messages())

    @patch('src.research_workflow.initialize_search_manager')
    @patch('src.research_workflow.GeminiModel')
    def test_process_user_input_search_manager_fails(self, mock_gemini_model, mock_search_manager):
        mock_search_manager.return_value = None
        mock_model_instance = MagicMock()
        mock_model_instance.chat.return_value = "This is a test response."
        mock_gemini_model.return_value = mock_model_instance
        
        user_input = "Test user input."
        response = process_user_input(user_input)
        self.assertEqual(response, "Search functionality is disabled.")
        mock_gemini_model.assert_not_called()
        mock_search_manager.assert_called_once()
        mock_model_instance.chat.assert_not_called()
        self.assertIn("Search manager initialization failed", self.get_log_messages())

    @patch('src.research_workflow.initialize_search_manager')
    @patch('src.research_workflow.GeminiModel')
    def test_process_user_input_unsupported_tool(self, mock_gemini_model, mock_search_manager):
        mock_search_manager.return_value.search.return_value = []
        mock_model_instance = MagicMock()
        mock_model_instance.chat.return_value = "Tool: Unsupported Tool, Query: | Test Query |"
        mock_gemini_model.return_value = mock_model_instance
        
        user_input = "Test user input."
        response = process_user_input(user_input)
        self.assertEqual(response, "Unsupported tool.")
        mock_gemini_model.assert_called_once()
        mock_search_manager.assert_called_once()
        mock_model_instance.chat.assert_called_once()
        self.assertIn("Unsupported tool requested", self.get_log_messages())

if __name__ == '__main__':
    unittest.main()

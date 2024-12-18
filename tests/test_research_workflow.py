import unittest
from unittest.mock import patch, MagicMock
from src.research_workflow import extract_tool_call, process_user_input
import logging

class TestResearchWorkflow(unittest.TestCase):

    def setUp(self):
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)
        self.handler = logging.StreamHandler()
        self.logger.addHandler(self.handler)

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def test_extract_tool_call_valid(self):
        text = "Tool: Web Search, Query: | What is the capital of France? |"
        expected = {"tool": "Web Search", "query": "What is the capital of France?"}
        self.assertEqual(extract_tool_call(text), expected)

    def test_extract_tool_call_no_match(self):
        text = "This is a regular text without a tool call."
        self.assertIsNone(extract_tool_call(text))

    def test_extract_tool_call_extra_spaces(self):
        text = "Tool:  Web Search  ,  Query:  |  What is the capital of France?  |  "
        expected = {"tool": "Web Search", "query": "What is the capital of France?"}
        self.assertEqual(extract_tool_call(text), expected)

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

if __name__ == '__main__':
    unittest.main()

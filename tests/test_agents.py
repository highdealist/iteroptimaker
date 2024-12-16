import unittest
import sys
import os
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List
import re

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.agent import BaseAgent
from src.agents.researcher_agent import ResearcherAgent
from src.models.model_manager import ModelManager
from src.tools.tool_manager import ToolManager
from src.tools.base_tool import ToolResult

class MockModel:
    def __init__(self, response: str = "Test response"):
        self.response = response

    def generate_response(self, prompt: str) -> str:
        return self.response

class MockTool(BaseAgent):
    def __init__(self, name: str, result: str = "Tool result"):
        self.name = name
        self.result = result
        self.description = "Mock tool description"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, result=self.result)

class TestBaseAgent(unittest.TestCase):
    def setUp(self):
        self.model_manager = MagicMock(spec=ModelManager)
        self.tool_manager = MagicMock(spec=ToolManager)
        self.model = MockModel()
        self.model_manager.get_model.return_value = self.model
        self.tool1 = MockTool(name="tool1")
        self.tool2 = MockTool(name="tool2", result="Another tool result")
        self.tool_manager.get_tool.side_effect = lambda name: {
            "tool1": self.tool1,
            "tool2": self.tool2,
        }.get(name)
        self.agent = BaseAgent(
            model_manager=self.model_manager,
            tool_manager=self.tool_manager,
            instruction="Test instruction",
            model_config={"model_type": "test_model"},
            name="test_agent"
        )

    def test_generate_response_no_tools(self):
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Test response")
        self.model_manager.get_model.assert_called_once_with("test_model")
        self.model.generate_response.assert_called_once()

    def test_generate_response_with_tool_call(self):
        self.model.response = "Response with tool call: <tool>tool1</tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_tool_call_and_args(self):
        self.model.response = 'Response with tool call: <tool name="tool2" arg1="value1" arg2="value2">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Another tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool2")
        self.tool1.execute.assert_not_called()
        self.tool2.execute.assert_called_once_with(arg1="value1", arg2="value2")

    def test_generate_response_with_multiple_tool_calls(self):
        self.model.response = 'Response with multiple tool calls: <tool>tool1</tool> and <tool name="tool2" arg="test">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with multiple tool calls: Tool result and Another tool result")
        self.tool_manager.get_tool.assert_any_call("tool1")
        self.tool_manager.get_tool.assert_any_call("tool2")
        self.assertEqual(self.tool1.execute.call_count, 1)
        self.tool2.execute.assert_called_once_with(arg="test")

    def test_generate_response_with_invalid_tool_call(self):
        self.model.response = "Response with invalid tool call: <tool>invalid_tool</tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with invalid tool call: <tool>invalid_tool</tool>")
        self.tool_manager.get_tool.assert_called_once_with("invalid_tool")
        self.tool_manager.get_tool.return_value.execute.assert_not_called()

    def test_generate_response_with_tool_call_and_text(self):
        self.model.response = "Some text before <tool>tool1</tool> and some text after"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Some text before Tool result and some text after")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_nested_tool_calls(self):
        self.model.response = "Nested <tool>tool1 <tool>tool2</tool></tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Nested Tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.tool_manager.get_tool.assert_not_called()
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_tool_call_and_special_characters(self):
        self.model.response = 'Response with tool call: <tool name="tool2" arg1="value with \\"quotes\\"" arg2="value with < > &">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Another tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool2")
        self.tool2.execute.assert_called_once_with(arg1='value with "quotes"', arg2='value with < > &')

    def test_construct_instructions(self):
        instructions = self.agent._construct_instructions()
        self.assertIn("Test instruction", instructions)
        self.assertIn("Available tools:", instructions)
        self.assertIn("tool1", instructions)
        self.assertIn("tool2", instructions)

    def test_parse_tool_args_empty(self):
        args = self.agent._parse_tool_args("")
        self.assertEqual(args, {})

    def test_parse_tool_args_single(self):
        args = self.agent._parse_tool_args('arg1="value1"')
        self.assertEqual(args, {"arg1": "value1"})

    def test_parse_tool_args_multiple(self):
        args = self.agent._parse_tool_args('arg1="value1" arg2="value2"')
        self.assertEqual(args, {"arg1": "value1", "arg2": "value2"})

    def test_parse_tool_args_with_spaces(self):
        args = self.agent._parse_tool_args('arg1="value with spaces"')
        self.assertEqual(args, {"arg1": "value with spaces"})

    def test_parse_tool_args_with_special_chars(self):
        args = self.agent._parse_tool_args('arg1="value with \\"quotes\\"" arg2="value with < > &"')
        self.assertEqual(args, {"arg1": 'value with "quotes"', "arg2": 'value with < > &'})

    def test_parse_list_safely(self):
        self.assertEqual(self.agent._parse_list_safely("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(self.agent._parse_list_safely("invalid"), [])

    def test_parse_dict_safely(self):
        self.assertEqual(self.agent._parse_dict_safely('{"key": "value"}'), {"key": "value"})
        self.assertEqual(self.agent._parse_dict_safely("invalid"), {})

class TestResearcherAgent(unittest.TestCase):
    def setUp(self):
        self.model_manager = MagicMock(spec=ModelManager)
        self.tool_manager = MagicMock(spec=ToolManager)
        self.model = MockModel()
        self.model_manager.get_model.return_value = self.model
        self.tool1 = MockTool(name="tool1")
        self.tool2 = MockTool(name="tool2", result="Another tool result")
        self.tool_manager.get_tool.side_effect = lambda name: {
            "tool1": self.tool1,
            "tool2": self.tool2,
        }.get(name)
        self.agent = ResearcherAgent(
            model_manager=self.model_manager,
            tool_manager=self.tool_manager,
            instruction="Test instruction",
            model_config={"model_type": "test_model"},
            name="test_researcher"
        )

    def test_generate_response_no_tools(self):
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Test response")
        self.model_manager.get_model.assert_called_once_with("test_model")
        self.model.generate_response.assert_called_once()

    def test_generate_response_with_tool_call(self):
        self.model.response = "Response with tool call: <tool>tool1</tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_tool_call_and_args(self):
        self.model.response = 'Response with tool call: <tool name="tool2" arg1="value1" arg2="value2">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Another tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool2")
        self.tool1.execute.assert_not_called()
        self.tool2.execute.assert_called_once_with(arg1="value1", arg2="value2")

    def test_generate_response_with_multiple_tool_calls(self):
        self.model.response = 'Response with multiple tool calls: <tool>tool1</tool> and <tool name="tool2" arg="test">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with multiple tool calls: Tool result and Another tool result")
        self.tool_manager.get_tool.assert_any_call("tool1")
        self.tool_manager.get_tool.assert_any_call("tool2")
        self.assertEqual(self.tool1.execute.call_count, 1)
        self.tool2.execute.assert_called_once_with(arg="test")

    def test_generate_response_with_invalid_tool_call(self):
        self.model.response = "Response with invalid tool call: <tool>invalid_tool</tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with invalid tool call: <tool>invalid_tool</tool>")
        self.tool_manager.get_tool.assert_called_once_with("invalid_tool")
        self.tool_manager.get_tool.return_value.execute.assert_not_called()

    def test_generate_response_with_tool_call_and_text(self):
        self.model.response = "Some text before <tool>tool1</tool> and some text after"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Some text before Tool result and some text after")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_nested_tool_calls(self):
        self.model.response = "Nested <tool>tool1 <tool>tool2</tool></tool>"
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Nested Tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool1")
        self.tool_manager.get_tool.assert_not_called()
        self.assertEqual(self.tool1.execute.call_count, 1)

    def test_generate_response_with_tool_call_and_special_characters(self):
        self.model.response = 'Response with tool call: <tool name="tool2" arg1="value with \\"quotes\\"" arg2="value with < > &">tool2</tool>'
        response = self.agent.generate_response("user input")
        self.assertEqual(response, "Response with tool call: Another tool result")
        self.tool_manager.get_tool.assert_called_once_with("tool2")
        self.tool2.execute.assert_called_once_with(arg1='value with "quotes"', arg2='value with < > &')

if __name__ == '__main__':
    unittest.main()

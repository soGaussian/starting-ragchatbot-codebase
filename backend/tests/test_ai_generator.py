"""
Tests for AIGenerator to validate tool calling and API interaction
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AIGenerator functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client"""
        return Mock()

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator with mock client"""
        with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
            generator = AIGenerator(api_key="test_key", model="claude-3-5-sonnet-20241022")
            generator.client = mock_anthropic_client
        return generator

    @pytest.fixture
    def sample_tools(self):
        """Sample tool definitions"""
        return [{
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"}
                },
                "required": ["query"]
            }
        }]

    @pytest.fixture
    def mock_tool_manager(self):
        """Create a mock ToolManager"""
        manager = Mock()
        manager.execute_tool.return_value = "Search results: Machine learning is..."
        return manager

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test basic response generation without tools"""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a test response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("What is AI?")

        assert result == "This is a test response"
        # Verify API was called correctly
        mock_anthropic_client.messages.create.assert_called_once()
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs['model'] == "claude-3-5-sonnet-20241022"
        assert call_kwargs['messages'][0]['content'] == "What is AI?"
        assert 'tools' not in call_kwargs
        print("\n✓ Basic response generation works")

    def test_generate_response_with_tools_no_use(self, ai_generator, mock_anthropic_client, sample_tools):
        """Test response when tools are available but not used"""
        mock_response = Mock()
        mock_response.content = [Mock(text="AI is artificial intelligence")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response(
            "What is AI?",
            tools=sample_tools
        )

        assert result == "AI is artificial intelligence"
        # Verify tools were passed
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert 'tools' in call_kwargs
        assert call_kwargs['tools'] == sample_tools
        print("\n✓ Response generation with available tools works")

    def test_generate_response_with_single_tool_round(self, ai_generator, mock_anthropic_client,
                                                      sample_tools, mock_tool_manager):
        """Test response when Claude uses tool once and completes"""
        # First response: tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "machine learning basics"}

        first_response = Mock()
        first_response.content = [mock_tool_use]
        first_response.stop_reason = "tool_use"

        # Second response: final answer (Claude stops after one search)
        second_response = Mock()
        second_response.content = [Mock(text="Machine learning is a subset of AI...")]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            "Tell me about machine learning",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )
        # Verify final response
        assert "Machine learning is a subset of AI" in result
        # Verify two API calls were made
        assert mock_anthropic_client.messages.create.call_count == 2
        print("\n✓ Single tool round flow works correctly")
        print(f"  Tool executed: {mock_tool_use.name}")
        print(f"  Final response: {result[:50]}...")

    def test_conversation_history_included(self, ai_generator, mock_anthropic_client):
        """Test that conversation history is included in the system prompt"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with context")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response(
            "Follow-up question",
            conversation_history=history
        )

        # Verify system prompt includes history
        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert history in call_kwargs['system']
        print("\n✓ Conversation history is included in system prompt")

    def test_api_parameters(self, ai_generator, mock_anthropic_client):
        """Test that correct API parameters are used"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("Test query")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['max_tokens'] == 800
        assert 'system' in call_kwargs
        print("\n✓ API parameters are correct:")
        print(f"  Temperature: {call_kwargs['temperature']}")
        print(f"  Max tokens: {call_kwargs['max_tokens']}")

    def test_handle_tool_execution_error(self, ai_generator, mock_anthropic_client,
                                        sample_tools, mock_tool_manager):
        """Test handling when tool execution returns an error"""
        # Mock tool returning error message
        mock_tool_manager.execute_tool.return_value = "Error: Database connection failed"

        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "test"}

        first_response = Mock()
        first_response.content = [mock_tool_use]
        first_response.stop_reason = "tool_use"

        second_response = Mock()
        second_response.content = [Mock(text="I apologize, there was an error")]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Tool should still be executed even if it returns an error
        mock_tool_manager.execute_tool.assert_called_once()
        # Second API call should include the error in tool results
        assert mock_anthropic_client.messages.create.call_count == 2
        print("\n✓ Tool execution error handling works")

    def test_system_prompt_content(self, ai_generator, mock_anthropic_client):
        """Test that system prompt contains expected guidance"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "end_turn"
        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("Test")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        system_prompt = call_kwargs['system']

        # Check for key elements in system prompt
        assert "course materials" in system_prompt.lower()
        assert "search" in system_prompt.lower() or "tool" in system_prompt.lower()
        print("\n✓ System prompt contains expected guidance")
        print(f"  System prompt length: {len(system_prompt)} chars")


    def test_two_sequential_tool_rounds(self, ai_generator, mock_anthropic_client,
                                       sample_tools, mock_tool_manager):
        """Test that Claude can use tools twice in sequence"""
        # First response: tool use (search course A)
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_123"
        mock_tool_use_1.input = {"query": "deep learning", "course_name": "ML Course"}

        first_response = Mock()
        first_response.content = [mock_tool_use_1]
        first_response.stop_reason = "tool_use"

        # Second response: tool use again (search course B)
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_456"
        mock_tool_use_2.input = {"query": "deep learning", "course_name": "AI Course"}

        second_response = Mock()
        second_response.content = [mock_tool_use_2]
        second_response.stop_reason = "tool_use"

        # Third response: final answer
        third_response = Mock()
        third_response.content = [Mock(text="Deep learning is covered in both courses...")]
        third_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            first_response, second_response, third_response
        ]

        # Mock different tool results
        mock_tool_manager.execute_tool.side_effect = [
            "ML Course content about deep learning...",
            "AI Course content about deep learning..."
        ]

        result = ai_generator.generate_response(
            "Compare deep learning in ML and AI courses",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify two tool executions
        assert mock_tool_manager.execute_tool.call_count == 2
        # Verify three API calls
        assert mock_anthropic_client.messages.create.call_count == 3
        # Verify final response
        assert "Deep learning is covered in both courses" in result
        print("\n✓ Two sequential tool rounds work correctly")
        print(f"  Tool calls: {mock_tool_manager.execute_tool.call_count}")
        print(f"  API calls: {mock_anthropic_client.messages.create.call_count}")

    def test_max_tool_rounds_enforced(self, ai_generator, mock_anthropic_client,
                                     sample_tools, mock_tool_manager):
        """Test that tool rounds are limited to MAX_TOOL_ROUNDS"""
        # First response: tool use
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_123"
        mock_tool_use_1.input = {"query": "test1"}

        first_response = Mock()
        first_response.content = [mock_tool_use_1]
        first_response.stop_reason = "tool_use"

        # Second response: tool use again
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_456"
        mock_tool_use_2.input = {"query": "test2"}

        second_response = Mock()
        second_response.content = [mock_tool_use_2]
        second_response.stop_reason = "tool_use"

        # Third response: forced completion (no tools available)
        third_response = Mock()
        third_response.content = [Mock(text="Final answer based on available information")]
        third_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [
            first_response, second_response, third_response
        ]

        result = ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify exactly 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2
        # Verify exactly 3 API calls
        assert mock_anthropic_client.messages.create.call_count == 3
        # Verify third call has no tools parameter (forced completion)
        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert 'tools' not in third_call_kwargs
        print("\n✓ Max tool rounds enforced correctly")
        print(f"  Tool executions: {mock_tool_manager.execute_tool.call_count}")
        print(f"  Third API call had no tools parameter")

    def test_tool_error_terminates_rounds(self, ai_generator, mock_anthropic_client,
                                         sample_tools, mock_tool_manager):
        """Test that tool errors prevent further tool rounds"""
        # Mock tool returning error
        mock_tool_manager.execute_tool.return_value = "Error: Database connection failed"

        # First response: tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "test"}

        first_response = Mock()
        first_response.content = [mock_tool_use]
        first_response.stop_reason = "tool_use"

        # Second response: error explanation (no tools available)
        second_response = Mock()
        second_response.content = [Mock(text="I encountered a database error")]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        result = ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify only one tool execution
        mock_tool_manager.execute_tool.assert_called_once()
        # Verify only 2 API calls (error terminates early)
        assert mock_anthropic_client.messages.create.call_count == 2
        # Verify second call has no tools parameter
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert 'tools' not in second_call_kwargs
        print("\n✓ Tool error terminates rounds correctly")

    def test_message_history_accumulates_across_rounds(self, ai_generator, mock_anthropic_client,
                                                      sample_tools, mock_tool_manager):
        """Test that messages accumulate correctly across rounds"""
        # Track message counts at each call
        message_counts = []

        def track_messages(*args, **kwargs):
            # Record the message count at this call
            message_counts.append(len(kwargs['messages']))
            # Return the appropriate response based on call count
            if len(message_counts) == 1:
                return first_response
            elif len(message_counts) == 2:
                return second_response
            else:
                return third_response

        # Setup two tool rounds
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_123"
        mock_tool_use_1.input = {"query": "test1"}

        first_response = Mock()
        first_response.content = [mock_tool_use_1]
        first_response.stop_reason = "tool_use"

        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_456"
        mock_tool_use_2.input = {"query": "test2"}

        second_response = Mock()
        second_response.content = [mock_tool_use_2]
        second_response.stop_reason = "tool_use"

        third_response = Mock()
        third_response.content = [Mock(text="Final answer")]
        third_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = track_messages

        ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        # Verify message counts at each call
        # First call: 1 message (user query)
        assert message_counts[0] == 1
        # Second call: 3 messages (user, assistant_tool_use, user_tool_result)
        assert message_counts[1] == 3
        # Third call: 5 messages (all previous + assistant_tool_use_2 + user_tool_result_2)
        assert message_counts[2] == 5
        print("\n✓ Message history accumulates correctly")
        print(f"  Round 1: {message_counts[0]} messages")
        print(f"  Round 2: {message_counts[1]} messages")
        print(f"  Round 3: {message_counts[2]} messages")

    def test_tools_parameter_preserved_in_intermediate_calls(self, ai_generator,
                                                            mock_anthropic_client,
                                                            sample_tools, mock_tool_manager):
        """Test that tools parameter is preserved until final call"""
        # Setup two tool rounds
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_123"
        mock_tool_use.input = {"query": "test"}

        first_response = Mock()
        first_response.content = [mock_tool_use]
        first_response.stop_reason = "tool_use"

        second_response = Mock()
        second_response.content = [Mock(text="Final answer")]
        second_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        ai_generator.generate_response(
            "Test query",
            tools=sample_tools,
            tool_manager=mock_tool_manager
        )

        call_args_list = mock_anthropic_client.messages.create.call_args_list

        # First call: should have tools
        first_call_kwargs = call_args_list[0][1]
        assert 'tools' in first_call_kwargs
        assert first_call_kwargs['tools'] == sample_tools
        assert 'tool_choice' in first_call_kwargs

        # Second call: should still have tools (Claude can decide to use them)
        second_call_kwargs = call_args_list[1][1]
        assert 'tools' in second_call_kwargs
        assert second_call_kwargs['tools'] == sample_tools
        print("\n✓ Tools parameter preserved in intermediate calls")


class TestAIGeneratorIntegration:
    """Integration tests that might need real API key"""

    def test_real_api_call_structure(self):
        """
        Test to validate the structure of real API calls.
        This test is skipped if no API key is available.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No ANTHROPIC_API_KEY available")

        generator = AIGenerator(api_key=api_key, model="claude-3-5-sonnet-20241022")

        # Test with a simple query that shouldn't need tools
        try:
            result = generator.generate_response("What is 2+2? Answer with just the number.")
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"\n✓ Real API call successful: {result}")
        except Exception as e:
            pytest.fail(f"Real API call failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

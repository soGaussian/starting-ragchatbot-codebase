"""
Tests for CourseSearchTool to validate search execution and error handling
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        return Mock()

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mock store"""
        return CourseSearchTool(mock_vector_store)

    def test_tool_definition(self, search_tool):
        """Test that tool definition is correctly formatted"""
        tool_def = search_tool.get_tool_definition()

        assert tool_def['name'] == 'search_course_content'
        assert 'description' in tool_def
        assert 'input_schema' in tool_def
        assert 'query' in tool_def['input_schema']['properties']
        assert tool_def['input_schema']['required'] == ['query']
        print("\n✓ Tool definition structure is correct")

    def test_execute_successful_search(self, search_tool, mock_vector_store):
        """Test successful search execution"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["This is content about machine learning from lesson 1."],
            metadata=[{
                'course_title': 'ML Course',
                'lesson_number': 1
            }],
            distances=[0.5],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        result = search_tool.execute(query="machine learning")

        assert isinstance(result, str)
        assert "ML Course" in result
        assert "This is content about machine learning" in result
        assert len(search_tool.last_sources) > 0
        print(f"\n✓ Successful search returns formatted results")
        print(f"  Result preview: {result[:100]}...")
        print(f"  Sources tracked: {len(search_tool.last_sources)}")

    def test_execute_with_error(self, search_tool, mock_vector_store):
        """Test search execution with error"""
        # Mock error results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test query")

        assert "Database connection failed" in result
        print(f"\n✓ Error handling works: {result}")

    def test_execute_no_results(self, search_tool, mock_vector_store):
        """Test search with no results"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result
        print(f"\n✓ No results message: {result}")

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test search with course name filter"""
        mock_results = SearchResults(
            documents=["Content from specific course"],
            metadata=[{'course_title': 'Specific Course', 'lesson_number': 1}],
            distances=[0.3],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        result = search_tool.execute(
            query="test query",
            course_name="Specific Course"
        )

        # Verify course name was passed to search
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Specific Course",
            lesson_number=None
        )
        assert "Specific Course" in result
        print(f"\n✓ Course filtering works")

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test search with lesson number filter"""
        mock_results = SearchResults(
            documents=["Content from lesson 3"],
            metadata=[{'course_title': 'Test Course', 'lesson_number': 3}],
            distances=[0.2],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"

        result = search_tool.execute(
            query="specific topic",
            lesson_number=3
        )

        mock_vector_store.search.assert_called_once_with(
            query="specific topic",
            course_name=None,
            lesson_number=3
        )
        assert "Lesson 3" in result
        print(f"\n✓ Lesson filtering works")

    def test_source_tracking(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked"""
        mock_results = SearchResults(
            documents=["Doc 1", "Doc 2"],
            metadata=[
                {'course_title': 'Course A', 'lesson_number': 1},
                {'course_title': 'Course A', 'lesson_number': 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2"
        ]
        mock_vector_store.search.return_value = mock_results

        result = search_tool.execute(query="test")

        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]['text'] == "Course A - Lesson 1"
        assert search_tool.last_sources[0]['url'] == "https://example.com/lesson1"
        assert search_tool.last_sources[1]['text'] == "Course A - Lesson 2"
        print(f"\n✓ Source tracking works correctly:")
        for source in search_tool.last_sources:
            print(f"  - {source['text']}: {source['url']}")

    def test_multiple_results_formatting(self, search_tool, mock_vector_store):
        """Test formatting of multiple search results"""
        mock_results = SearchResults(
            documents=[
                "First relevant document about the topic.",
                "Second document with more information.",
                "Third document from different lesson."
            ],
            metadata=[
                {'course_title': 'Course X', 'lesson_number': 1},
                {'course_title': 'Course X', 'lesson_number': 1},
                {'course_title': 'Course X', 'lesson_number': 2}
            ],
            distances=[0.1, 0.2, 0.3],
            error=None
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = None

        result = search_tool.execute(query="topic")

        # Verify all documents are included
        assert "First relevant document" in result
        assert "Second document" in result
        assert "Third document" in result
        # Verify structure
        assert "[Course X - Lesson 1]" in result
        assert "[Course X - Lesson 2]" in result
        print(f"\n✓ Multiple results formatted correctly")
        print(f"  Result length: {len(result)} chars")


class TestToolManager:
    """Test ToolManager functionality"""

    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = Mock()
        tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object", "properties": {}}
        }
        tool.execute.return_value = "Tool executed successfully"
        return tool

    def test_register_tool(self, tool_manager, mock_tool):
        """Test tool registration"""
        tool_manager.register_tool(mock_tool)

        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]['name'] == 'test_tool'
        print("\n✓ Tool registration works")

    def test_execute_tool(self, tool_manager, mock_tool):
        """Test tool execution through manager"""
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", param1="value1")

        mock_tool.execute.assert_called_once_with(param1="value1")
        assert result == "Tool executed successfully"
        print("\n✓ Tool execution through manager works")

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing a tool that doesn't exist"""
        result = tool_manager.execute_tool("nonexistent_tool")

        assert "not found" in result
        print(f"\n✓ Nonexistent tool handling: {result}")

    def test_get_last_sources(self, tool_manager, mock_tool):
        """Test retrieving sources from tools"""
        # Add last_sources attribute to mock
        mock_tool.last_sources = [{"text": "Source 1", "url": "http://example.com"}]
        tool_manager.register_tool(mock_tool)

        sources = tool_manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]['text'] == "Source 1"
        print("\n✓ Source retrieval from tools works")

    def test_reset_sources(self, tool_manager, mock_tool):
        """Test resetting sources across all tools"""
        mock_tool.last_sources = [{"text": "Source 1"}]
        tool_manager.register_tool(mock_tool)

        tool_manager.reset_sources()

        assert mock_tool.last_sources == []
        print("\n✓ Source reset works")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

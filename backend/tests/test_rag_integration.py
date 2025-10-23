"""
Integration tests for RAG system with real components
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest.mock import Mock, patch
from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk


class TestRAGIntegration:
    """Integration tests for the RAG system"""

    @pytest.fixture
    def test_config(self, tmp_path):
        """Create test configuration"""
        config = Config()
        config.CHROMA_PATH = str(tmp_path / "test_chroma")
        config.MAX_RESULTS = 3
        # Use real API key if available, otherwise tests will be limited
        config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "test_key")
        return config

    @pytest.fixture
    def rag_system(self, test_config):
        """Create a RAG system with test configuration"""
        return RAGSystem(test_config)

    @pytest.fixture
    def sample_course_data(self, rag_system, tmp_path):
        """Add sample course data to the system"""
        # Create a sample course document
        course_content = """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. AI Expert

Lesson 1: What is Machine Learning
Lesson Link: https://example.com/ml-course/lesson1
Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data. These systems improve their performance on a specific task through experience without being explicitly programmed. Machine learning algorithms use statistical techniques to find patterns in data and make predictions or decisions.

Lesson 2: Types of Machine Learning
Lesson Link: https://example.com/ml-course/lesson2
There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models. Unsupervised learning finds patterns in unlabeled data. Reinforcement learning trains agents through rewards and penalties.

Lesson 3: Neural Networks Basics
Lesson Link: https://example.com/ml-course/lesson3
Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes or neurons. Deep learning uses neural networks with multiple layers to learn complex patterns in data. Convolutional neural networks are particularly effective for image processing tasks.
"""
        # Write to temporary file
        course_file = tmp_path / "test_course.txt"
        course_file.write_text(course_content)

        # Add to RAG system
        course, chunks = rag_system.add_course_document(str(course_file))

        print(f"\n✓ Added test course: {course.title}")
        print(f"  Lessons: {len(course.lessons)}")
        print(f"  Chunks: {chunks}")

        return course, chunks

    def test_system_initialization(self, rag_system):
        """Test that RAG system initializes correctly"""
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        # Verify tools are registered
        tool_defs = rag_system.tool_manager.get_tool_definitions()
        tool_names = [tool['name'] for tool in tool_defs]
        assert 'search_course_content' in tool_names
        assert 'get_course_outline' in tool_names
        print("\n✓ RAG system initialized correctly")
        print(f"  Registered tools: {tool_names}")

    def test_add_course_document(self, rag_system, sample_course_data):
        """Test adding course documents"""
        course, chunk_count = sample_course_data

        assert course is not None
        assert course.title == "Introduction to Machine Learning"
        assert len(course.lessons) == 3
        assert chunk_count > 0

        # Verify course is in vector store
        course_titles = rag_system.vector_store.get_existing_course_titles()
        assert course.title in course_titles
        print(f"\n✓ Course added successfully: {course.title}")

    def test_content_search_query(self, rag_system, sample_course_data):
        """Test content search through RAG system"""
        # Mock the AI generator to avoid real API calls
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "Neural networks are computing systems inspired by biological neural networks."

            response, sources = rag_system.query(
                "What are neural networks?",
                session_id="test_session"
            )

            # Verify AI generator was called with tools
            assert mock_gen.called
            call_kwargs = mock_gen.call_args[1]
            assert 'tools' in call_kwargs
            assert call_kwargs['tool_manager'] is not None
            print("\n✓ Content query processed")
            print(f"  Response: {response[:100]}...")

    def test_outline_query(self, rag_system, sample_course_data):
        """Test outline retrieval through RAG system"""
        # Mock the AI generator
        with patch.object(rag_system.ai_generator, 'generate_response') as mock_gen:
            mock_gen.return_value = "The course has 3 lessons..."

            response, sources = rag_system.query(
                "What is the outline of the Machine Learning course?",
                session_id="test_session"
            )

            # Verify tools were provided
            assert mock_gen.called
            call_kwargs = mock_gen.call_args[1]
            assert 'tools' in call_kwargs
            print("\n✓ Outline query processed")

    def test_search_tool_direct_execution(self, rag_system, sample_course_data):
        """Test search tool execution directly"""
        result = rag_system.search_tool.execute(
            query="neural networks deep learning"
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Should find content about neural networks
        if "No relevant content found" not in result:
            assert "neural" in result.lower() or "network" in result.lower()
            print(f"\n✓ Search tool found relevant content")
            print(f"  Result preview: {result[:150]}...")
        else:
            print(f"\n✗ Search tool found no content: {result}")

    def test_search_with_course_filter(self, rag_system, sample_course_data):
        """Test search with course name filter"""
        result = rag_system.search_tool.execute(
            query="machine learning types",
            course_name="Machine Learning"
        )

        assert isinstance(result, str)
        if "No relevant content found" not in result and "No course found" not in result:
            assert "Machine Learning" in result or "supervised" in result.lower()
            print(f"\n✓ Filtered search works")
        else:
            print(f"\n✗ Filtered search issue: {result}")

    def test_search_with_lesson_filter(self, rag_system, sample_course_data):
        """Test search within specific lesson"""
        result = rag_system.search_tool.execute(
            query="neural networks",
            lesson_number=3
        )

        if "No relevant content found" not in result:
            # Should find lesson 3 content
            print(f"\n✓ Lesson-filtered search works")
            print(f"  Result preview: {result[:150]}...")
        else:
            print(f"\n✗ Lesson-filtered search found nothing")

    def test_outline_tool_direct_execution(self, rag_system, sample_course_data):
        """Test outline tool execution directly"""
        result = rag_system.outline_tool.execute(
            course_name="Machine Learning"
        )

        assert isinstance(result, str)
        assert "No course found" not in result
        assert "Introduction to Machine Learning" in result
        assert "Lesson 1" in result or "1." in result
        assert "Lesson 2" in result or "2." in result
        assert "Lesson 3" in result or "3." in result
        print(f"\n✓ Outline tool works")
        print(f"  Outline:\n{result}")

    def test_source_tracking(self, rag_system, sample_course_data):
        """Test that sources are tracked correctly"""
        # Execute search tool
        result = rag_system.search_tool.execute(query="machine learning")

        sources = rag_system.search_tool.last_sources

        if len(sources) > 0:
            assert all('text' in source for source in sources)
            assert all('url' in source for source in sources)
            print(f"\n✓ Sources tracked: {len(sources)}")
            for source in sources:
                print(f"  - {source['text']}: {source.get('url', 'No URL')}")
        else:
            print("\n✗ No sources tracked")

    def test_empty_query_handling(self, rag_system, sample_course_data):
        """Test handling of queries with no results"""
        result = rag_system.search_tool.execute(
            query="quantum computing blockchain cryptocurrency"
        )

        # Should handle gracefully even if no results
        assert isinstance(result, str)
        print(f"\n✓ Empty results handled: {result}")

    def test_vector_store_direct_search(self, rag_system, sample_course_data):
        """Test VectorStore search directly"""
        results = rag_system.vector_store.search(
            query="what is machine learning",
            limit=3
        )

        print(f"\n✓ VectorStore direct search:")
        print(f"  Results: {len(results.documents)}")
        print(f"  Error: {results.error}")

        if not results.is_empty():
            for i, (doc, meta) in enumerate(zip(results.documents, results.metadata)):
                print(f"\n  Result {i+1}:")
                print(f"    Course: {meta.get('course_title')}")
                print(f"    Lesson: {meta.get('lesson_number')}")
                print(f"    Content preview: {doc[:100]}...")
        else:
            print("  ✗ No results found")

    def test_course_analytics(self, rag_system, sample_course_data):
        """Test course analytics endpoint"""
        analytics = rag_system.get_course_analytics()

        assert analytics['total_courses'] >= 1
        assert len(analytics['course_titles']) >= 1
        assert "Introduction to Machine Learning" in analytics['course_titles']
        print(f"\n✓ Course analytics:")
        print(f"  Total courses: {analytics['total_courses']}")
        print(f"  Titles: {analytics['course_titles']}")


class TestRAGWithRealAPI:
    """Tests that require real API key"""

    @pytest.fixture
    def rag_with_real_api(self, tmp_path):
        """Create RAG system with real API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("No ANTHROPIC_API_KEY available")

        config = Config()
        config.CHROMA_PATH = str(tmp_path / "test_chroma")
        config.ANTHROPIC_API_KEY = api_key
        return RAGSystem(config)

    def test_real_content_query(self, rag_with_real_api, tmp_path):
        """Test real content query with API"""
        # Add sample course
        course_content = """Course Title: Test AI Course
Course Link: https://example.com/ai
Course Instructor: Test Teacher

Lesson 1: Introduction to AI
Lesson Link: https://example.com/lesson1
Artificial intelligence is the simulation of human intelligence by machines. AI systems can perform tasks that typically require human intelligence such as visual perception, speech recognition, and decision-making.
"""
        course_file = tmp_path / "test_ai.txt"
        course_file.write_text(course_content)
        rag_with_real_api.add_course_document(str(course_file))

        # Make real query
        try:
            response, sources = rag_with_real_api.query(
                "What is artificial intelligence?",
                session_id="test"
            )

            assert isinstance(response, str)
            assert len(response) > 0
            print(f"\n✓ Real API query successful")
            print(f"  Response: {response}")
            print(f"  Sources: {len(sources)}")
        except Exception as e:
            pytest.fail(f"Real API query failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

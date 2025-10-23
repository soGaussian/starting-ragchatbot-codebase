"""
Tests for VectorStore to validate data indexing and retrieval
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from vector_store import VectorStore
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test VectorStore functionality"""

    @pytest.fixture
    def vector_store(self, tmp_path):
        """Create a temporary VectorStore for testing"""
        test_db_path = str(tmp_path / "test_chroma")
        store = VectorStore(
            chroma_path=test_db_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        return store

    @pytest.fixture
    def sample_course(self):
        """Create a sample course for testing"""
        course = Course(
            title="Test Course on Machine Learning",
            course_link="https://example.com/ml-course",
            instructor="Dr. Test",
            lessons=[
                Lesson(lesson_number=1, title="Introduction to ML", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/lesson2")
            ]
        )
        return course

    @pytest.fixture
    def sample_chunks(self):
        """Create sample course chunks"""
        chunks = [
            CourseChunk(
                content="Lesson 1 content: Machine learning is a subset of artificial intelligence.",
                course_title="Test Course on Machine Learning",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Neural networks are computing systems inspired by biological neural networks.",
                course_title="Test Course on Machine Learning",
                lesson_number=2,
                chunk_index=1
            ),
            CourseChunk(
                content="Course Test Course on Machine Learning Lesson 2 content: Deep learning uses multiple layers in neural networks.",
                course_title="Test Course on Machine Learning",
                lesson_number=2,
                chunk_index=2
            )
        ]
        return chunks

    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata to catalog"""
        vector_store.add_course_metadata(sample_course)

        # Verify course was added
        course_titles = vector_store.get_existing_course_titles()
        assert sample_course.title in course_titles
        assert vector_store.get_course_count() == 1

    def test_add_course_content(self, vector_store, sample_course, sample_chunks):
        """Test adding course content chunks"""
        # First add metadata
        vector_store.add_course_metadata(sample_course)

        # Then add content
        vector_store.add_course_content(sample_chunks)

        # Verify we can search the content
        results = vector_store.search(
            query="machine learning artificial intelligence",
            limit=5
        )

        assert not results.is_empty()
        assert len(results.documents) > 0
        print(f"\n✓ Found {len(results.documents)} results for ML query")

    def test_course_name_resolution(self, vector_store, sample_course, sample_chunks):
        """Test fuzzy course name matching"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Test partial match
        results = vector_store.search(
            query="neural networks",
            course_name="Machine Learning",  # Partial match
            limit=5
        )

        if results.error:
            print(f"\n✗ Course resolution error: {results.error}")
        else:
            print(f"\n✓ Fuzzy course name match worked, found {len(results.documents)} results")

        assert not results.error

    def test_search_with_lesson_filter(self, vector_store, sample_course, sample_chunks):
        """Test searching within a specific lesson"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        # Search lesson 2 specifically
        results = vector_store.search(
            query="neural networks",
            lesson_number=2,
            limit=5
        )

        assert not results.is_empty()
        # All results should be from lesson 2
        for metadata in results.metadata:
            assert metadata.get('lesson_number') == 2
        print(f"\n✓ Lesson filtering works: {len(results.documents)} results from lesson 2")

    def test_search_with_course_and_lesson_filter(self, vector_store, sample_course, sample_chunks):
        """Test combined course and lesson filtering"""
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_chunks)

        results = vector_store.search(
            query="deep learning",
            course_name="Test Course",
            lesson_number=2,
            limit=5
        )

        if not results.is_empty():
            print(f"\n✓ Combined filters work: {len(results.documents)} results")
            for doc, meta in zip(results.documents, results.metadata):
                print(f"  - Lesson {meta.get('lesson_number')}: {doc[:100]}...")
        else:
            print(f"\n✗ No results with combined filters")

    def test_get_lesson_link(self, vector_store, sample_course):
        """Test retrieving lesson links"""
        vector_store.add_course_metadata(sample_course)

        link = vector_store.get_lesson_link(sample_course.title, 1)
        assert link == "https://example.com/lesson1"
        print(f"\n✓ Lesson link retrieval works: {link}")

    def test_get_course_outline(self, vector_store, sample_course):
        """Test retrieving complete course outline"""
        vector_store.add_course_metadata(sample_course)

        outline = vector_store.get_course_outline("Machine Learning")

        assert outline is not None
        assert outline['course_title'] == sample_course.title
        assert outline['course_link'] == sample_course.course_link
        assert outline['instructor'] == sample_course.instructor
        assert len(outline['lessons']) == 2
        print(f"\n✓ Course outline retrieval works:")
        print(f"  Title: {outline['course_title']}")
        print(f"  Lessons: {len(outline['lessons'])}")

    def test_metadata_consistency(self, vector_store, sample_chunks):
        """Test that chunk metadata is consistent"""
        vector_store.add_course_content(sample_chunks)

        # Search and verify metadata structure
        results = vector_store.search(query="machine learning", limit=10)

        for metadata in results.metadata:
            assert 'course_title' in metadata
            assert 'chunk_index' in metadata
            # lesson_number can be None for non-lesson content
            print(f"\n✓ Metadata structure: course={metadata.get('course_title')}, "
                  f"lesson={metadata.get('lesson_number')}, chunk={metadata.get('chunk_index')}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

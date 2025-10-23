"""
Simple test runner without pytest - directly tests the components
"""
import sys
import os
import tempfile
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk


def test_vector_store():
    """Test VectorStore basic functionality"""
    print("\n" + "="*70)
    print("TESTING VECTOR STORE")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create store
        store = VectorStore(
            chroma_path=tmpdir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )

        # Create test course
        course = Course(
            title="Test ML Course",
            course_link="https://example.com/ml",
            instructor="Dr. Test",
            lessons=[
                Lesson(lesson_number=1, title="Intro to ML", lesson_link="https://example.com/l1"),
                Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/l2")
            ]
        )

        # Test adding metadata
        print("\n[TEST] Adding course metadata...")
        try:
            store.add_course_metadata(course)
            courses = store.get_existing_course_titles()
            assert course.title in courses, f"Course not found in: {courses}"
            print("✓ PASS: Course metadata added successfully")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test adding content
        print("\n[TEST] Adding course content...")
        chunks = [
            CourseChunk(
                content="Lesson 1 content: Machine learning is a subset of AI.",
                course_title="Test ML Course",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Course Test ML Course Lesson 2 content: Neural networks use layers of neurons.",
                course_title="Test ML Course",
                lesson_number=2,
                chunk_index=1
            )
        ]

        try:
            store.add_course_content(chunks)
            print("✓ PASS: Content added successfully")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test search
        print("\n[TEST] Searching for content...")
        try:
            results = store.search("machine learning artificial intelligence")
            print(f"  Found {len(results.documents)} results")
            if results.error:
                print(f"  Error: {results.error}")
            if results.documents:
                print(f"  First result: {results.documents[0][:100]}...")
                print(f"  Metadata: {results.metadata[0]}")
                print("✓ PASS: Search executed successfully")
            else:
                print("✗ FAIL: No results found")
                return False
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test fuzzy course name match
        print("\n[TEST] Fuzzy course name matching...")
        try:
            results = store.search("neural networks", course_name="ML Course")
            if results.error:
                print(f"  ✗ FAIL: {results.error}")
                return False
            print(f"  Found {len(results.documents)} results with fuzzy match")
            print("✓ PASS: Fuzzy matching works")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test outline retrieval
        print("\n[TEST] Getting course outline...")
        try:
            outline = store.get_course_outline("ML Course")
            if outline:
                print(f"  Course: {outline['course_title']}")
                print(f"  Lessons: {len(outline['lessons'])}")
                print("✓ PASS: Outline retrieval works")
            else:
                print("✗ FAIL: No outline found")
                return False
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

    return True


def test_search_tool():
    """Test CourseSearchTool"""
    print("\n" + "="*70)
    print("TESTING SEARCH TOOL")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        store = VectorStore(tmpdir, "all-MiniLM-L6-v2", 5)
        course = Course(
            title="AI Fundamentals",
            course_link="https://example.com/ai",
            lessons=[Lesson(lesson_number=1, title="What is AI", lesson_link="https://example.com/l1")]
        )
        store.add_course_metadata(course)

        chunks = [
            CourseChunk(
                content="Lesson 1 content: Artificial intelligence involves creating smart machines.",
                course_title="AI Fundamentals",
                lesson_number=1,
                chunk_index=0
            )
        ]
        store.add_course_content(chunks)

        # Create tool
        tool = CourseSearchTool(store)

        # Test tool definition
        print("\n[TEST] Tool definition...")
        try:
            tool_def = tool.get_tool_definition()
            assert tool_def['name'] == 'search_course_content'
            assert 'query' in tool_def['input_schema']['properties']
            print("✓ PASS: Tool definition correct")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test execution
        print("\n[TEST] Tool execution...")
        try:
            result = tool.execute(query="artificial intelligence")
            print(f"  Result length: {len(result)} chars")
            print(f"  Result preview: {result[:150]}...")
            if "No relevant content found" not in result:
                print("✓ PASS: Tool execution successful")
            else:
                print(f"✗ FAIL: No results: {result}")
                return False
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

        # Test source tracking
        print("\n[TEST] Source tracking...")
        try:
            sources = tool.last_sources
            print(f"  Sources tracked: {len(sources)}")
            if sources:
                print(f"  First source: {sources[0]}")
                print("✓ PASS: Sources tracked")
            else:
                print("⚠ WARNING: No sources tracked")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            return False

    return True


def test_outline_tool():
    """Test CourseOutlineTool"""
    print("\n" + "="*70)
    print("TESTING OUTLINE TOOL")
    print("="*70)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(tmpdir, "all-MiniLM-L6-v2", 5)
        course = Course(
            title="Deep Learning Basics",
            course_link="https://example.com/dl",
            instructor="Dr. Network",
            lessons=[
                Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/l1"),
                Lesson(lesson_number=2, title="CNNs", lesson_link="https://example.com/l2"),
                Lesson(lesson_number=3, title="RNNs", lesson_link="https://example.com/l3")
            ]
        )
        store.add_course_metadata(course)

        tool = CourseOutlineTool(store)

        # Test tool definition
        print("\n[TEST] Outline tool definition...")
        try:
            tool_def = tool.get_tool_definition()
            assert tool_def['name'] == 'get_course_outline'
            print("✓ PASS: Outline tool definition correct")
        except Exception as e:
            print(f"✗ FAIL: {e}")
            return False

        # Test execution
        print("\n[TEST] Outline tool execution...")
        try:
            result = tool.execute(course_name="Deep Learning")
            print(f"  Result:\n{result}")
            # Check for key elements in the outline
            if ("Deep Learning Basics" in result and
                ("3)" in result or "lessons (3)" in result.lower()) and
                "Introduction" in result):
                print("✓ PASS: Outline retrieved correctly")
            else:
                print(f"✗ FAIL: Unexpected result")
                return False
        except Exception as e:
            print(f"✗ FAIL: {e}")
            traceback.print_exc()
            return False

    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("RAG SYSTEM COMPONENT TESTS")
    print("="*70)

    results = []

    # Run tests
    results.append(("VectorStore", test_vector_store()))
    results.append(("SearchTool", test_search_tool()))
    results.append(("OutlineTool", test_outline_tool()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} test suites passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

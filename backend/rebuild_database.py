"""
Rebuild the ChromaDB database with fixed code
This will clear existing data and re-index all courses
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from rag_system import RAGSystem

def main():
    print("="*70)
    print("REBUILDING RAG DATABASE")
    print("="*70)

    print("\n⚠️  This will delete all existing vector data and rebuild from scratch.")
    print(f"Database path: {config.CHROMA_PATH}")
    print(f"Documents path: ../docs")

    # Create RAG system
    print("\n[1] Initializing RAG system...")
    rag = RAGSystem(config)

    # Clear existing data
    print("\n[2] Clearing existing database...")
    rag.vector_store.clear_all_data()
    print("✓ Database cleared")

    # Reload all documents
    print("\n[3] Loading documents from ../docs...")
    docs_path = "../docs"
    if not os.path.exists(docs_path):
        print(f"✗ ERROR: Documents folder not found at {docs_path}")
        return 1

    courses, chunks = rag.add_course_folder(docs_path, clear_existing=False)

    print(f"\n✓ Successfully indexed:")
    print(f"  - {courses} courses")
    print(f"  - {chunks} content chunks")

    # Verify
    print("\n[4] Verifying database...")
    analytics = rag.get_course_analytics()
    print(f"✓ Verification:")
    print(f"  - Total courses in DB: {analytics['total_courses']}")
    print(f"  - Course titles:")
    for title in analytics['course_titles']:
        print(f"    • {title}")

    # Test search
    print("\n[5] Testing search functionality...")
    try:
        result = rag.search_tool.execute(query="machine learning")
        if "No relevant content found" not in result:
            print("✓ Search is working!")
            print(f"  Sample result: {result[:150]}...")
        else:
            print("⚠️  Search returned no results (may be expected if no ML content)")
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return 1

    print("\n" + "="*70)
    print("✅ DATABASE REBUILD COMPLETE!")
    print("="*70)
    print("\nThe server should now work correctly with content queries.")
    print("You can restart the server or it will pick up the new data automatically.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

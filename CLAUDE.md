# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Set up environment
# Create .env file with: ANTHROPIC_API_KEY=your_key_here
```

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application serves:
- Web interface: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Architecture Overview

### RAG System Design

This is a tool-based Retrieval-Augmented Generation (RAG) system where the AI uses tools to search course materials, rather than manually passing context. The key architectural pattern is:

1. **User query** → RAGSystem
2. **RAGSystem** provides search tools to AIGenerator
3. **AIGenerator** (Claude) decides when/how to use search tools
4. **Search results** automatically tracked and returned with response

### Core Components

**RAGSystem** (`backend/rag_system.py`)
- Main orchestrator that wires together all components
- Initializes: DocumentProcessor, VectorStore, AIGenerator, SessionManager, ToolManager
- Handles document ingestion and query processing
- Manages tool registration and source tracking

**AIGenerator** (`backend/ai_generator.py`)
- Wraps Anthropic's Claude API with tool support
- Implements agentic loop: message → tool use → tool results → final response
- System prompt emphasizes: search only for course-specific questions, one search max, direct answers without meta-commentary

**VectorStore** (`backend/vector_store.py`)
- Uses ChromaDB with two collections:
  - `course_catalog`: Course metadata (title, instructor, lessons) for course name resolution
  - `course_content`: Chunked lesson content for semantic search
- Implements unified search interface with smart course name matching (fuzzy matching via vector similarity)
- Supports filtering by course title and/or lesson number

**DocumentProcessor** (`backend/document_processor.py`)
- Parses structured course documents with expected format:
  ```
  Course Title: [title]
  Course Link: [url]
  Course Instructor: [name]

  Lesson N: [title]
  Lesson Link: [url]
  [content]
  ```
- Chunks text using sentence-based approach with configurable size/overlap
- Adds context to chunks (course title + lesson number) for better retrieval

**ToolManager & CourseSearchTool** (`backend/search_tools.py`)
- Implements tool abstraction for Claude's function calling
- `CourseSearchTool` provides semantic search with optional course/lesson filters
- Tracks sources from searches for UI display
- Tool definition exposes: query (required), course_name (optional), lesson_number (optional)

**SessionManager** (`backend/session_manager.py`)
- Manages conversation history with configurable `MAX_HISTORY` limit
- Maintains context across multi-turn conversations
- Session IDs created automatically if not provided

### Data Flow

**Document Ingestion:**
```
File → DocumentProcessor.process_course_document()
     → (Course, List[CourseChunk])
     → VectorStore.add_course_metadata() + add_course_content()
     → ChromaDB collections
```

**Query Processing:**
```
User query → RAGSystem.query()
           → AIGenerator.generate_response() with tools
           → Claude decides to use search_course_content tool
           → ToolManager.execute_tool()
           → CourseSearchTool.execute()
           → VectorStore.search() (resolves course name, filters, searches)
           → Results formatted and returned to Claude
           → Claude synthesizes final answer
           → Response + sources returned to user
```

### Configuration

All settings in `backend/config.py`:
- `ANTHROPIC_MODEL`: Currently "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2" (SentenceTransformers)
- `CHUNK_SIZE`: 800 characters
- `CHUNK_OVERLAP`: 100 characters
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges

### Key Design Decisions

**Tool-based RAG**: The system uses Claude's tool calling rather than stuffing context into prompts. This allows Claude to decide when to search and what parameters to use.

**Two-tier vector storage**: Separating course metadata from content enables fuzzy course name matching before content search, improving search accuracy.

**Chunk context augmentation**: Each chunk includes course title and lesson number prefix to improve retrieval relevance when chunks are embedded.

**Incremental loading**: On startup, the system loads documents from `../docs` folder, skipping courses already in the vector store (based on title matching).

**Stateless sessions**: Session history is maintained in-memory only; sessions don't persist across server restarts.

## Project Structure

```
backend/
  ├── app.py              # FastAPI application & endpoints
  ├── rag_system.py       # Main orchestrator
  ├── ai_generator.py     # Claude API wrapper
  ├── vector_store.py     # ChromaDB interface
  ├── document_processor.py # Document parsing & chunking
  ├── search_tools.py     # Tool definitions & management
  ├── session_manager.py  # Conversation history
  ├── models.py           # Pydantic models
  └── config.py           # Configuration

frontend/
  ├── index.html          # Web UI
  ├── script.js           # Frontend logic
  └── style.css           # Styling

docs/                     # Course documents (.txt, .pdf, .docx)
chroma_db/               # ChromaDB persistence (gitignored)
```

## Important Notes

- The system expects `.env` file with `ANTHROPIC_API_KEY` in the root directory
- Course documents in `docs/` are loaded automatically on startup
- The vector store persists in `./chroma_db` and survives restarts
- When adding new courses, the system checks for duplicates by title to avoid re-indexing
- The search tool uses vector similarity to match course names, so exact spelling isn't required
- make sure to use uv to manage all dependencies
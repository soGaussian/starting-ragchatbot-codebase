# Bug Analysis and Proposed Fixes

## Issue: Content Queries Return "Query Failed"

### Root Cause Identified

**ChromaDB Metadata `None` Value Error**

Location: `backend/vector_store.py`, lines 145-160

#### The Problem:
ChromaDB cannot store `None` values in metadata fields. When courses or lessons have optional fields set to `None` (instructor, course_link, lesson_link), the `add()` method fails with:
```
TypeError: argument 'metadatas': failed to extract enum MetadataValue ('Bool | Int | Float | Str')
```

#### Impact:
1. Course metadata fails to be added to the vector store
2. Course content chunks may fail to be added
3. Search queries find no data because indexing failed
4. The error is silent - no clear error message surfaces to the user
5. Results in "query failed" or empty results for all content queries

### Evidence from Tests:

```
[TEST] Adding course metadata...
✓ PASS: Course metadata added successfully     # Works when all fields have values

[TEST] Searching for content...
✓ PASS: Search executed successfully           # Search works when data is indexed

# But then:
TypeError when instructor=None or course_link=None # FAILS with None values
```

## Proposed Fixes

### Fix 1: Filter None Values from Metadata (RECOMMENDED)

**File:** `backend/vector_store.py`

**Line 150-160:** Update `add_course_metadata()` method:

```python
# Before (BROKEN):
self.course_catalog.add(
    documents=[course_text],
    metadatas=[{
        "title": course.title,
        "instructor": course.instructor,  # Can be None!
        "course_link": course.course_link,  # Can be None!
        "lessons_json": json.dumps(lessons_metadata),
        "lesson_count": len(course.lessons)
    }],
    ids=[course.title]
)

# After (FIXED):
# Build metadata dict and filter out None values
metadata = {
    "title": course.title,
    "lessons_json": json.dumps(lessons_metadata),
    "lesson_count": len(course.lessons)
}

# Only add non-None optional fields
if course.instructor is not None:
    metadata["instructor"] = course.instructor
if course.course_link is not None:
    metadata["course_link"] = course.course_link

self.course_catalog.add(
    documents=[course_text],
    metadatas=[metadata],
    ids=[course.title]
)
```

**Line 142-148:** Update lessons metadata building:

```python
# Before (BROKEN):
lessons_metadata.append({
    "lesson_number": lesson.lesson_number,
    "lesson_title": lesson.title,
    "lesson_link": lesson.lesson_link  # Can be None!
})

# After (FIXED):
lesson_meta = {
    "lesson_number": lesson.lesson_number,
    "lesson_title": lesson.title
}
if lesson.lesson_link is not None:
    lesson_meta["lesson_link"] = lesson.lesson_link

lessons_metadata.append(lesson_meta)
```

### Fix 2: Document Processing Chunk Context Inconsistency

**File:** `backend/document_processor.py`

**Issue:** Inconsistent context prefixing between first lessons and last lesson:
- Lines 186-188: First chunks get `"Lesson {current_lesson} content: {chunk}"`
- Line 234: Last lesson chunks get `"Course {course_title} Lesson {current_lesson} content: {chunk}"`

**Fix:** Make context prefixing consistent:

```python
# Line 186 - Make consistent with last lesson formatting
chunk_with_context = f"Course {course.title} Lesson {current_lesson} content: {chunk}"
```

OR remove course title from both to simplify:

```python
# Line 186 and 234 - Use simple format
chunk_with_context = f"Lesson {current_lesson} content: {chunk}"
```

### Fix 3: Better Error Handling and Logging

**File:** `backend/vector_store.py`

Add try-catch with better error messages:

```python
def add_course_metadata(self, course: Course):
    """Add course information to the catalog for semantic search"""
    import json

    try:
        course_text = course.title

        # Build lessons metadata (filter None values)
        lessons_metadata = []
        for lesson in course.lessons:
            lesson_meta = {
                "lesson_number": lesson.lesson_number,
                "lesson_title": lesson.title
            }
            if lesson.lesson_link is not None:
                lesson_meta["lesson_link"] = lesson.lesson_link
            lessons_metadata.append(lesson_meta)

        # Build course metadata (filter None values)
        metadata = {
            "title": course.title,
            "lessons_json": json.dumps(lessons_metadata),
            "lesson_count": len(course.lessons)
        }

        if course.instructor is not None:
            metadata["instructor"] = course.instructor
        if course.course_link is not None:
            metadata["course_link"] = course.course_link

        self.course_catalog.add(
            documents=[course_text],
            metadatas=[metadata],
            ids=[course.title]
        )

        print(f"✓ Successfully added course metadata: {course.title}")

    except Exception as e:
        print(f"✗ Error adding course metadata for '{course.title}': {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to surface the error
```

## Testing Recommendations

After applying fixes:

1. **Test with None values:**
   ```python
   course = Course(
       title="Test Course",
       course_link=None,  # Test None
       instructor=None,   # Test None
       lessons=[
           Lesson(lesson_number=1, title="Test", lesson_link=None)  # Test None
       ]
   )
   ```

2. **Test existing course data:**
   - Verify all 4 existing courses can be re-indexed
   - Check if any have None values that were failing silently

3. **Test search after fix:**
   - Content queries should return results
   - Sources should be tracked correctly
   - Links should handle None gracefully

## Priority

**CRITICAL** - This bug completely breaks content search functionality. Should be fixed immediately.

## Estimated Impact

After fix:
- ✓ Course indexing will succeed even with optional fields
- ✓ Content queries will return results
- ✓ Search tool will execute correctly
- ✓ Source tracking will work
- ✓ System will be stable for production use

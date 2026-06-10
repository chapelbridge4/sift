"""
Tests for structure-aware Markdown chunking.
"""

import pytest
from app.services.text_splitter import MarkdownChunker, RecursiveCharacterTextSplitter


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_split_by_headings(self):
        """Test that ## headings become chunk boundaries."""
        text = """# Main Title

Some intro text.

## Section One

Content of section one.

## Section Two

Content of section two.
"""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        # Should have chunks for each section
        contents = [c.content for c in chunks]

        # Section headings should be preserved
        assert any("Section One" in c for c in contents)
        assert any("Section Two" in c for c in contents)

    def test_code_block_atomic_preservation(self):
        """Test that triple-backtick code blocks are kept as atomic units."""
        text = """Some text before.

```python
def hello():
    print("Hello, World!")
    return True
```

Some text after."""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        # At least one chunk should contain the entire code block
        contents = [c.content for c in chunks]
        code_chunks = [c for c in contents if "```python" in c and "Hello" in c]
        assert len(code_chunks) > 0

    def test_code_block_not_split_inside(self):
        """Test that code blocks are never split in the middle."""
        text = """Start.

```python
line1
line2
line3
line4
line5
```

End."""

        # Very small chunk size to force splitting - but code should stay atomic
        chunker = MarkdownChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(text)

        # No chunk should contain a partial code block
        for chunk in chunks:
            content = chunk.content
            # If it starts with ``` it should also end with ```
            if content.startswith("```"):
                assert content.strip().endswith("```"), f"Code block was split: {content[:50]}..."

    def test_heading_path_propagation(self):
        """Test that heading_path metadata tracks parent headings."""
        text = """# Title

## Section A

Content under A.

### Subsection A1

Content under A1.

## Section B

Content under B.
"""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        # Find the chunk containing "Content under A1"
        for chunk in chunks:
            if "Content under A1" in chunk.content:
                # Should have Section A and Subsection A1 in heading path
                assert "Section A" in chunk.metadata.get("heading_path", [])
                assert "Subsection A1" in chunk.metadata.get("heading_path", [])
                break

    def test_chunk_type_classification(self):
        """Test chunk_type classification: paragraph, code, heading, table."""
        text = """## Heading Chunk

Regular paragraph.

```python
print("code")
```

| col1 | col2 |
|------|------|
| a    | b    |
"""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        chunk_types = {c.metadata.get("chunk_type") for c in chunks}

        assert "heading" in chunk_types
        assert "code" in chunk_types
        assert "table" in chunk_types
        assert "paragraph" in chunk_types

    def test_paragraph_splitting(self):
        """Test that double newlines create paragraph boundaries."""
        text = """Paragraph one.

Paragraph two.

Paragraph three."""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        contents = [c.content for c in chunks]
        assert any("Paragraph one" in c for c in contents)
        assert any("Paragraph two" in c for c in contents)
        assert any("Paragraph three" in c for c in contents)

    def test_empty_paragraphs_skipped(self):
        """Test that empty segments are not included as chunks."""
        text = """First paragraph.


Second paragraph."""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        contents = [c.content for c in chunks]
        assert not any(c.strip() == "" for c in contents)

    def test_position_metadata(self):
        """Test that position index is set correctly."""
        text = """## Section

Content one.

Content two."""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        positions = [c.metadata.get("position") for c in chunks]
        assert positions == list(range(len(chunks)))

    def test_no_headings_returns_paragraphs(self):
        """Test that plain text without headings gets paragraph chunking."""
        text = """This is a plain text document.

It has multiple paragraphs.

And should be chunked by paragraph boundaries."""
        chunker = MarkdownChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(text)

        # Should still get chunked, not an error
        assert len(chunks) > 0
        # All should be paragraph type since no headings
        for chunk in chunks:
            if chunk.content.strip():
                assert chunk.metadata.get("chunk_type") in ("paragraph", "code", "table")


class TestRecursiveCharacterTextSplitter:
    """Tests for fallback RecursiveCharacterTextSplitter."""

    def test_split_by_double_newline(self):
        """Test paragraph splitting on double newlines."""
        text = """First paragraph.

Second paragraph.

Third paragraph."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_text(text)

        contents = [c["content"] for c in chunks]
        assert any("First paragraph" in c for c in contents)
        assert any("Second paragraph" in c for c in contents)

    def test_large_segment_splitting(self):
        """Test that segments larger than chunk_size get split."""
        text = "A" * 1000  # Very long text
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_text(text)

        # Should produce multiple chunks
        assert len(chunks) > 1

    def test_metadata_fields(self):
        """Test that metadata includes required fields."""
        text = """Some text."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "chunk_type" in chunk["metadata"]
            assert "heading_path" in chunk["metadata"]
            assert "position" in chunk["metadata"]
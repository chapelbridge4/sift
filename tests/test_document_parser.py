"""
Focused tests for document parser chunking behavior.
Tests structure-aware chunking quality on text fixtures.
"""

import unittest
from pathlib import Path

from app.services.document_parser import DocumentParser


class TestStructureAwareChunking(unittest.TestCase):
    """Test structure-aware chunking on text fixtures."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_single_short_paragraph(self):
        """Test that a single short paragraph becomes one chunk."""
        text = "This is a short paragraph with less than 512 characters."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content.strip(), text)

    def test_multiple_paragraphs_split_cleanly(self):
        """Test that double-newline separated paragraphs are preserved."""
        text = "First paragraph here.\n\nSecond paragraph here."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertGreaterEqual(len(chunks), 1)
        first_chunk_content = chunks[0].content
        self.assertIn("First paragraph", first_chunk_content)

    def test_metadata_includes_document_id(self):
        """Test that chunks have a stable document_id."""
        text = "Test content."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertEqual(len(chunks), 1)
        self.assertIn("document_id", chunks[0].metadata)
        self.assertIsInstance(chunks[0].metadata["document_id"], str)
        self.assertEqual(len(chunks[0].metadata["document_id"]), 16)

    def test_metadata_includes_text_boundaries(self):
        """Test that chunks have text_start and text_end positions."""
        text = "A" * 1000
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("text_start", chunk.metadata)
            self.assertIn("text_end", chunk.metadata)
            self.assertGreaterEqual(chunk.metadata["text_end"], chunk.metadata["text_start"])

    def test_metadata_includes_chunk_char_length(self):
        """Test that chunk metadata includes character length."""
        text = "Test content here."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].metadata["chunk_char_length"],
            len(chunks[0].content)
        )

    def test_metadata_total_chunks_set_after_creation(self):
        """Test that total_chunks is set after all chunks are created."""
        text = "A" * 2000
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        for chunk in chunks:
            self.assertEqual(chunk.metadata["total_chunks"], len(chunks))

    def test_chunk_index_sequential(self):
        """Test that chunk indices are sequential."""
        text = "A" * 2000
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["chunk_index"], i)

    def test_large_paragraph_split_at_sentence_boundary(self):
        """Test that large paragraphs split at sentence boundaries when possible."""
        text = ("This is sentence one. " * 100) + "\n\nExtra content."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), self.parser.chunk_size + 100)

    def test_empty_text_returns_no_chunks(self):
        """Test that empty text returns empty list."""
        chunks = self.parser._create_chunks("", "/tmp/test.txt")
        self.assertEqual(len(chunks), 0)

    def test_whitespace_only_text_returns_no_chunks(self):
        """Test that whitespace-only text returns empty list."""
        chunks = self.parser._create_chunks("   \n\n   \t   ", "/tmp/test.txt")
        self.assertEqual(len(chunks), 0)


class TestChunkingFromFixtures(unittest.TestCase):
    """Test chunking behavior using benchmark fixture content."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_hippocampus_article_chunks(self):
        """Test chunking on hippocampus article content."""
        text = """The hippocampus is a major component of the brain of humans and other vertebrates.
It belongs to the limbic system and plays important roles in the consolidation of information
from short-term memory to long-term memory, and in spatial memory that enables navigation.
The hippocampus is located under the cerebral cortex in the medial temporal lobe."""

        chunks = self.parser._create_chunks(text, "/tmp/neuroscience_101.txt")

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("hippocampus", chunk.content.lower())

    def test_ai_article_chunks(self):
        """Test chunking on AI article content."""
        text = """Artificial Intelligence (AI) is a branch of computer science that aims to create
intelligent machines that can think and act like humans. Machine learning is a subset of AI
that allows systems to learn from data without being explicitly programmed. Deep learning
uses neural networks with many layers to model complex patterns in data."""

        chunks = self.parser._create_chunks(text, "/tmp/ai_intro.txt")

        self.assertGreater(len(chunks), 0)
        found_topics = False
        for chunk in chunks:
            content_lower = chunk.content.lower()
            if any(topic in content_lower for topic in ["artificial intelligence", "machine learning", "deep learning"]):
                found_topics = True
                break
        self.assertTrue(found_topics, "Expected AI topics not found in chunks")

    def test_python_article_chunks(self):
        """Test chunking on Python article content."""
        text = """Python is a high-level programming language known for its readability and
simplicity. It supports multiple programming paradigms including procedural, object-oriented,
and functional programming. Python's extensive standard library and package ecosystem make
it popular for web development, data analysis, machine learning, and automation tasks."""

        chunks = self.parser._create_chunks(text, "/tmp/python_overview.txt")

        self.assertGreater(len(chunks), 0)
        found_python = False
        for chunk in chunks:
            if "python" in chunk.content.lower():
                found_python = True
                break
        self.assertTrue(found_python)

    def test_metadata_source_file_preserved(self):
        """Test that source_file metadata is correctly set."""
        text = "Some content here."
        chunks = self.parser._create_chunks(text, "/tmp/my_document.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["source_file"], "my_document.txt")

    def test_metadata_file_path_does_not_leak_absolute_path(self):
        """Test that public metadata does not expose local absolute paths."""
        text = "Some content here."
        chunks = self.parser._create_chunks(text, "/tmp/private/my_document.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["file_path"], "my_document.txt")
        self.assertFalse(Path(chunks[0].metadata["file_path"]).is_absolute())

    def test_metadata_file_type_preserved(self):
        """Test that file_type metadata is correctly set."""
        text = "Some content here."
        chunks = self.parser._create_chunks(text, "/tmp/document.pdf")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["file_type"], ".pdf")


class TestDocumentIdStability(unittest.TestCase):
    """Test that document IDs are stable for the same file."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_same_file_same_id(self):
        """Test that same file path produces same document ID."""
        text = "Content"
        chunks1 = self.parser._create_chunks(text, "/tmp/test.txt")
        chunks2 = self.parser._create_chunks(text, "/tmp/test.txt")

        self.assertEqual(chunks1[0].metadata["document_id"],
                        chunks2[0].metadata["document_id"])

    def test_different_files_different_ids(self):
        """Test that different files produce different document IDs."""
        text = "Same content"
        chunks1 = self.parser._create_chunks(text, "/tmp/file1.txt")
        chunks2 = self.parser._create_chunks(text, "/tmp/file2.txt")

        self.assertNotEqual(chunks1[0].metadata["document_id"],
                           chunks2[0].metadata["document_id"])


class TestChunkBoundaries(unittest.TestCase):
    """Test that chunk boundaries are cleaner than naive character splitting."""

    def setUp(self):
        self.parser = DocumentParser()

    def test_chunks_are_not_excessively_long(self):
        """Test that no chunk significantly exceeds chunk_size."""
        text = "word " * 500
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        for chunk in chunks:
            self.assertLessEqual(
                len(chunk.content),
                self.parser.chunk_size + 200,
                f"Chunk too long: {len(chunk.content)}"
            )

    def test_consecutive_chunks_have_overlap_info(self):
        """Test that chunks have proper text boundary metadata for overlap detection."""
        text = "A" * 1000 + " B" * 1000 + " C" * 1000
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                self.assertLess(
                    chunks[i + 1].metadata["text_start"],
                    chunks[i].metadata["text_end"],
                    "Next chunk start should be before current chunk end (overlap expected)"
                )

    def test_consecutive_chunks_do_not_abut(self):
        """Test that consecutive chunks have proper overlap, not abutting boundaries."""
        text = "A" * 800 + " B" * 800 + " C" * 800
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        if len(chunks) > 1:
            for i in range(len(chunks) - 1):
                self.assertLess(
                    chunks[i + 1].metadata["text_start"],
                    chunks[i].metadata["text_end"],
                    "Consecutive chunks should overlap, not abut"
                )

    def test_chunk_spans_align_with_original_text(self):
        """Test that chunk spans are aligned with the original text offsets."""
        text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
        chunks = self.parser._create_chunks(text, "/tmp/test.txt")

        for chunk in chunks:
            extracted = text[chunk.metadata["text_start"]:chunk.metadata["text_end"]]
            self.assertEqual(
                extracted,
                chunk.content,
                f"Chunk span [{chunk.metadata['text_start']}:{chunk.metadata['text_end']}] "
                f"should extract content matching chunk.content"
            )

    def test_large_segments_preserve_overlap(self):
        """Test that large segments produce chunks with real overlap."""
        parser = DocumentParser()
        text = "word " * 600
        chunks = parser._create_chunks(text, "/tmp/test.txt")

        if len(chunks) > 1:
            first_chunk_end = chunks[0].metadata["text_end"]
            second_chunk_start = chunks[1].metadata["text_start"]
            self.assertLess(
                second_chunk_start,
                first_chunk_end,
                "Consecutive chunks should overlap"
            )
            overlap_amount = first_chunk_end - second_chunk_start
            self.assertGreater(
                overlap_amount,
                0,
                "Overlap should be meaningful"
            )

    def test_overlapped_large_segment_tail_is_non_empty(self):
        """Test that the final tail chunk of an overlapped large segment has non-empty content."""
        parser = DocumentParser()
        text = "word " * 600
        chunks = parser._create_chunks(text, "/tmp/test.txt")

        self.assertGreater(len(chunks), 1)
        last_chunk = chunks[-1]
        self.assertGreater(
            len(last_chunk.content),
            0,
            f"Final tail chunk should be non-empty, got content={repr(last_chunk.content)}"
        )


if __name__ == '__main__':
    unittest.main()

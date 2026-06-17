"""
Structure-aware text splitting for Markdown and plain text documents.
Provides heading-aware, code-block-preserving chunking with rich metadata.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ChunkResult:
    """A single chunk with content and metadata."""
    content: str
    metadata: Dict[str, Any]


class MarkdownChunker:
    """
    Splits text into chunks with awareness of Markdown structure:
    - ## headings become chunk boundaries with heading_path metadata
    - Triple-backtick code blocks are kept atomic (never split inside)
    - Paragraphs (\\n\\n) are preferred boundaries
    - Falls back to RecursiveCharacterTextSplitter for unstructured text
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[ChunkResult]:
        """
        Main entry point: split text into structured chunks.

        Args:
            text: Input text (Markdown or plain)

        Returns:
            List of ChunkResult with content and metadata
        """
        # First, extract code blocks as atomic units
        code_blocks, text_without_code = self._extract_code_blocks(text)

        # Split remaining text by structure (headings, paragraphs)
        chunks = []
        current_headings: List[str] = []

        # Split by ## headings first
        sections = self._split_by_headings(text_without_code)

        for section_text, section_headings in sections:
            # Update heading path
            if section_headings:
                current_headings = section_headings

            # Split by paragraphs within each section
            paragraphs = self._split_by_paragraphs(section_text)

            for para_text in paragraphs:
                if not para_text.strip():
                    continue

                para_text = para_text.strip()

                # Determine chunk type
                chunk_type = self._classify_chunk(para_text)

                # If it's a heading section that wasn't split by ##, handle it
                if chunk_type == "heading" and not para_text.startswith("##"):
                    chunk_type = "paragraph"

                # Check if chunk is too large
                if len(para_text) <= self.chunk_size * 1.5:
                    chunks.append(
                        ChunkResult(
                            content=para_text,
                            metadata={
                                "chunk_type": chunk_type,
                                "heading_path": list(current_headings),
                                "position": len(chunks),
                            }
                        )
                    )
                else:
                    # Fallback to recursive splitting for large chunks
                    sub_chunks = self._split_large_chunk(
                        para_text, current_headings, len(chunks), chunk_type
                    )
                    chunks.extend(sub_chunks)

        # Re-insert code blocks as their own chunks
        chunks = self._reinsert_code_blocks(chunks, code_blocks)

        # Update positions after code block insertion
        for i, chunk in enumerate(chunks):
            chunk.metadata["position"] = i

        return chunks

    def _extract_code_blocks(
        self, text: str
    ) -> Tuple[List[Tuple[str, int, int]], str]:
        """
        Extract triple-backtick code blocks as atomic units.
        Returns (code_blocks, text_with_placeholder).

        Each code block is (code_content, start_offset, end_offset).
        """
        code_blocks: List[Tuple[str, int, int]] = []
        pattern = re.compile(r'```[\w]*\n[\s\S]*?```', re.MULTILINE)

        for match in pattern.finditer(text):
            code_blocks.append((match.group(), match.start(), match.end()))

        # Replace code blocks with placeholders
        text_without_code = pattern.sub(r'\n[CODECHUNK]\n', text)

        return code_blocks, text_without_code

    def _reinsert_code_blocks(
        self,
        chunks: List[ChunkResult],
        code_blocks: List[Tuple[str, int, int]]
    ) -> List[ChunkResult]:
        """Re-insert code blocks as separate chunks at appropriate positions."""
        if not code_blocks:
            return chunks

        result: List[ChunkResult] = []
        code_idx = 0

        for chunk in chunks:
            # Insert any code blocks that appear before this text chunk
            while code_idx < len(code_blocks) and code_blocks[code_idx][2] <= chunk.metadata.get("_text_offset", 0):
                code_content, start, end = code_blocks[code_idx]
                result.append(
                    ChunkResult(
                        content=code_content,
                        metadata={
                            "chunk_type": "code",
                            "heading_path": chunk.metadata.get("heading_path", []),
                            "position": len(result),
                        }
                    )
                )
                code_idx += 1

            # Add the text chunk
            result.append(chunk)

        # Add any remaining code blocks
        while code_idx < len(code_blocks):
            code_content, start, end = code_blocks[code_idx]
            result.append(
                ChunkResult(
                    content=code_content,
                    metadata={
                        "chunk_type": "code",
                        "heading_path": [],
                        "position": len(result),
                    }
                )
            )
            code_idx += 1

        return result

    def _split_by_headings(
        self, text: str
    ) -> List[Tuple[str, List[str]]]:
        """
        Split text by ## headings. Each heading becomes a section.
        Returns list of (section_text, section_heading_path) tuples.

        The heading text itself is NOT included in section_text - it's tracked
        in the heading_path for that section's chunks.
        """
        heading_pattern = re.compile(r'^(#{2,})\s+(.+)$', re.MULTILINE)

        sections: List[Tuple[str, List[str]]] = []
        current_headings: List[str] = []
        prev_end = 0

        for match in heading_pattern.finditer(text):
            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Content between previous heading (or start) and this heading
            if match.start() > prev_end:
                content = text[prev_end:match.start()]
                if content.strip():
                    sections.append((content.strip(), list(current_headings)))

            # Include the heading line itself as content under its own heading path
            heading_line = match.group(0).strip()
            sections.append((heading_line, list(current_headings)))

            # Update heading path for level 2+ headings
            if level == 2:
                if current_headings:
                    current_headings[-1] = heading_text
                else:
                    current_headings.append(heading_text)
            elif level > 2:
                # For subheadings (### etc.), append to path
                if current_headings and current_headings[-1] == heading_text:
                    pass  # same heading, no change
                else:
                    current_headings.append(heading_text)

            prev_end = match.end()

        # Add remaining content after last heading
        if prev_end < len(text):
            remaining = text[prev_end:]
            if remaining.strip():
                sections.append((remaining.strip(), list(current_headings)))

        if not sections:
            sections.append((text.strip(), []))

        return sections

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines (paragraph boundaries)."""
        parts = re.split(r'\n\n+', text)
        return [p.strip() for p in parts if p.strip()]

    def _classify_chunk(self, text: str) -> str:
        """Classify chunk type based on content."""
        text_stripped = text.strip()

        if text_stripped.startswith("##"):
            return "heading"
        if text_stripped.startswith("```"):
            return "code"
        if self._is_table(text_stripped):
            return "table"
        return "paragraph"

    def _is_table(self, text: str) -> bool:
        """Simple heuristic: lines with multiple pipes and consistent column count."""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False

        # Check if lines look like table rows (multiple pipes)
        pipe_counts = [line.count('|') for line in lines if line.strip()]
        if not pipe_counts or max(pipe_counts) < 2:
            return False

        # Check consistency
        return len(set(pipe_counts)) == 1

    def _split_large_chunk(
        self,
        text: str,
        heading_path: List[str],
        start_position: int,
        chunk_type: str
    ) -> List[ChunkResult]:
        """
        Fallback: split large chunk using sentence/word boundaries.
        Used when a paragraph exceeds chunk_size * 1.5.
        """
        chunks = []
        char_pos = 0
        position = start_position
        prev_pos = -1  # sentinel: first chunk never triggers the no-progress guard below

        while char_pos < len(text):
            remaining = text[char_pos:]
            if len(remaining) <= self.chunk_size:
                chunks.append(
                    ChunkResult(
                        content=remaining.strip(),
                        metadata={
                            "chunk_type": chunk_type,
                            "heading_path": heading_path,
                            "position": position,
                        }
                    )
                )
                break

            # Take chunk_size characters
            chunk_text = remaining[:self.chunk_size]

            # Find best break point
            break_point = self._find_best_break_point(chunk_text)

            if break_point > self.chunk_size * 0.4:
                actual_chunk = chunk_text[:break_point + 1]
                actual_end = char_pos + break_point + 1
            else:
                actual_chunk = chunk_text.strip()
                actual_end = char_pos + self.chunk_size

            chunks.append(
                ChunkResult(
                    content=actual_chunk.strip(),
                    metadata={
                        "chunk_type": chunk_type,
                        "heading_path": heading_path,
                        "position": position,
                    }
                )
            )

            position += 1
            char_pos = actual_end - self.chunk_overlap
            if char_pos <= prev_pos:  # overlap would stall forward progress → skip it
                char_pos = actual_end
            prev_pos = actual_end

        return chunks

    def _find_best_break_point(self, chunk_text: str) -> int:
        """
        Find best break point within chunk_text.
        Prefers sentence > paragraph > word > comma boundaries.
        """
        # Sentence endings
        sentence_ends = list(re.finditer(r'[.!?]+\s+', chunk_text))
        if sentence_ends:
            last = sentence_ends[-1]
            if last.start() > self.chunk_size * 0.5:
                return last.start()

        # Newlines
        newline_pos = chunk_text.rfind('\n')
        if newline_pos > self.chunk_size * 0.5:
            return newline_pos

        # Word boundaries
        space_pos = chunk_text.rfind(' ')
        if space_pos > self.chunk_size * 0.5:
            return space_pos

        # Commas as last resort
        comma_pos = chunk_text.rfind(',')
        if comma_pos > self.chunk_size * 0.7:
            return comma_pos

        return -1


class RecursiveCharacterTextSplitter:
    """
    Fallback splitter for non-Markdown text.
    Splits on double newlines, then single newlines, then words.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text with recursive character splitting.
        Returns list of dicts with content and metadata.
        """
        chunks = []
        segments = self._split_by_double_newline(text)

        for segment in segments:
            if len(segment) <= self.chunk_size * 1.5:
                if segment.strip():
                    chunks.append({
                        "content": segment.strip(),
                        "metadata": {
                            "chunk_type": "paragraph",
                            "heading_path": [],
                            "position": len(chunks),
                        }
                    })
            else:
                sub_chunks = self._split_large_segment(segment, len(chunks))
                chunks.extend(sub_chunks)

        # Update positions
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["position"] = i

        return chunks

    def _split_by_double_newline(self, text: str) -> List[str]:
        """Split by \\n\\n boundaries."""
        parts = re.split(r'\n\n+', text)
        return [p.strip() for p in parts if p.strip()]

    def _split_large_segment(
        self, segment: str, start_position: int
    ) -> List[Dict[str, Any]]:
        """Split large segment recursively."""
        chunks = []
        char_pos = 0
        position = start_position

        while char_pos < len(segment):
            remaining = segment[char_pos:]
            if len(remaining) <= self.chunk_size:
                chunks.append({
                    "content": remaining.strip(),
                    "metadata": {
                        "chunk_type": "paragraph",
                        "heading_path": [],
                        "position": position,
                    }
                })
                break

            chunk_text = remaining[:self.chunk_size]
            break_point = self._find_break_point(chunk_text)

            if break_point > self.chunk_size * 0.4:
                actual_chunk = chunk_text[:break_point + 1]
                actual_end = char_pos + break_point + 1
            else:
                actual_chunk = chunk_text.strip()
                actual_end = char_pos + self.chunk_size

            chunks.append({
                "content": actual_chunk.strip(),
                "metadata": {
                    "chunk_type": "paragraph",
                    "heading_path": [],
                    "position": position,
                }
            })

            position += 1
            char_pos = actual_end - self.chunk_overlap
            if char_pos <= 0:
                char_pos = actual_end

        return chunks

    def _find_break_point(self, chunk_text: str) -> int:
        """Find best break point using sentence/word boundaries."""
        sentence_ends = list(re.finditer(r'[.!?]+\s+', chunk_text))
        if sentence_ends:
            last = sentence_ends[-1]
            if last.start() > self.chunk_size * 0.5:
                return last.start()

        newline_pos = chunk_text.rfind('\n')
        if newline_pos > self.chunk_size * 0.5:
            return newline_pos

        space_pos = chunk_text.rfind(' ')
        if space_pos > self.chunk_size * 0.5:
            return space_pos

        comma_pos = chunk_text.rfind(',')
        if comma_pos > self.chunk_size * 0.7:
            return comma_pos

        return -1
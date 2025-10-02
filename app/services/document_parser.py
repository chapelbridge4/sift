"""
Document parsing service for multiple file formats.
Supports PDF, TXT, DOC, DOCX, XLS, XLSX with async batch processing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import PyPDF2
import fitz  # PyMuPDF
from docx import Document
import openpyxl
import chardet
from loguru import logger

from app.config import get_settings
from app.utils.async_helpers import AsyncBatchProcessor, chunks


class DocumentChunk:
    """Represents a chunk of text from a document."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int = 0
    ):
        self.content = content
        self.metadata = metadata
        self.chunk_index = chunk_index


class DocumentParser:
    """Parse documents of various formats into text chunks."""

    def __init__(self):
        self.settings = get_settings()
        self.chunk_size = self.settings.CHUNK_SIZE
        self.chunk_overlap = self.settings.CHUNK_OVERLAP
        self.batch_processor = AsyncBatchProcessor(
            batch_size=self.settings.BATCH_SIZE,
            max_workers=self.settings.MAX_WORKERS
        )

    async def parse_files(self, file_paths: List[str]) -> List[DocumentChunk]:
        """
        Parse multiple files and return document chunks.

        Args:
            file_paths: List of file paths to parse

        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Starting to parse {len(file_paths)} files")

        # Parse files in batches using thread pool
        all_chunks = await self.batch_processor.process_in_batches(
            items=file_paths,
            process_func=self._parse_single_file_sync,
            use_threads=True
        )

        # Flatten the list of chunks
        flattened_chunks = []
        for file_chunks in all_chunks:
            if file_chunks:  # Skip None or empty results
                flattened_chunks.extend(file_chunks)

        logger.info(f"Parsed {len(flattened_chunks)} total chunks from {len(file_paths)} files")
        return flattened_chunks

    def _parse_single_file_sync(self, file_path: str) -> List[DocumentChunk]:
        """
        Parse a single file (synchronous function for thread pool).

        Args:
            file_path: Path to the file

        Returns:
            List of DocumentChunk objects
        """
        try:
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                logger.error(f"File not found: {file_path}")
                return []

            # Check file size
            file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
            if file_size_mb > self.settings.MAX_FILE_SIZE_MB:
                logger.error(f"File too large ({file_size_mb:.2f}MB): {file_path}")
                return []

            extension = file_path_obj.suffix.lower()
            logger.debug(f"Parsing {extension} file: {file_path}")

            # Route to appropriate parser
            if extension == '.pdf':
                text = self._parse_pdf(file_path)
            elif extension == '.txt':
                text = self._parse_txt(file_path)
            elif extension in ['.doc', '.docx']:
                text = self._parse_docx(file_path)
            elif extension in ['.xls', '.xlsx']:
                text = self._parse_excel(file_path)
            else:
                logger.warning(f"Unsupported file format: {extension}")
                return []

            if not text or not text.strip():
                logger.warning(f"No text extracted from: {file_path}")
                return []

            # Create chunks from text
            chunks = self._create_chunks(text, file_path)
            logger.info(f"Created {len(chunks)} chunks from {file_path_obj.name}")

            return chunks

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            return []

    def _parse_pdf(self, file_path: str) -> str:
        """Parse PDF file using PyMuPDF (faster and more accurate)."""
        try:
            text_parts = []
            doc = fitz.open(file_path)

            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")

            doc.close()
            return "\n\n".join(text_parts)

        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {str(e)}")
            # Fallback to PyPDF2
            try:
                text_parts = []
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        if text.strip():
                            text_parts.append(f"[Page {page_num}]\n{text}")
                return "\n\n".join(text_parts)
            except Exception as e2:
                logger.error(f"Both PDF parsers failed: {str(e2)}")
                return ""

    def _parse_txt(self, file_path: str) -> str:
        """Parse text file with encoding detection."""
        try:
            # Detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'

            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                return file.read()

        except Exception as e:
            logger.error(f"Error parsing text file: {str(e)}")
            return ""

    def _parse_docx(self, file_path: str) -> str:
        """Parse DOCX file."""
        try:
            doc = Document(file_path)
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        text_parts.append(row_text)

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error parsing DOCX file: {str(e)}")
            return ""

    def _parse_excel(self, file_path: str) -> str:
        """Parse Excel file (XLS/XLSX)."""
        try:
            workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            text_parts = []

            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"[Sheet: {sheet_name}]")

                for row in sheet.iter_rows(values_only=True):
                    row_text = ' | '.join(str(cell) if cell is not None else '' for cell in row)
                    if row_text.strip():
                        text_parts.append(row_text)

            workbook.close()
            return "\n".join(text_parts)

        except Exception as e:
            logger.error(f"Error parsing Excel file: {str(e)}")
            return ""

    def _create_chunks(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.

        Args:
            text: Full text to chunk
            source_file: Source file path

        Returns:
            List of DocumentChunk objects
        """
        chunks_list = []
        file_path_obj = Path(source_file)

        # Simple character-based chunking with overlap
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence end
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                last_space = chunk_text.rfind(' ')

                break_point = max(last_period, last_newline, last_space)
                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk size
                    chunk_text = chunk_text[:break_point + 1]
                    end = start + break_point + 1

            chunk_text = chunk_text.strip()
            if chunk_text:
                metadata = {
                    'source_file': file_path_obj.name,
                    'file_path': str(file_path_obj.absolute()),
                    'file_type': file_path_obj.suffix.lower(),
                    'chunk_index': chunk_index,
                }

                chunks_list.append(
                    DocumentChunk(
                        content=chunk_text,
                        metadata=metadata,
                        chunk_index=chunk_index
                    )
                )
                chunk_index += 1

            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)

        return chunks_list

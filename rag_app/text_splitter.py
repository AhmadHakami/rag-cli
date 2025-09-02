from __future__ import annotations

from typing import List, Dict, Any
from .interfaces import DocumentChunk
from .pdf_loader import load_pdf_pages_structured
from .interfaces import IChatLLM


def split_pages_into_chunks(doc_id: str, pages: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[DocumentChunk]:
    """Legacy splitter for backward compatibility."""
    chunks: List[DocumentChunk] = []
    for i, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(DocumentChunk(doc_id=doc_id, page_number=i, text=chunk_text))
            if end == len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks


def split_structured_pdf_into_chunks(doc_id: str, pdf_path: str, llm: IChatLLM, max_chunk_size: int = 1000) -> List[DocumentChunk]:
    """Intelligent LLM-based chunking: use LLM to split text into logical chunks."""
    structured_pages = load_pdf_pages_structured(pdf_path)
    chunks: List[DocumentChunk] = []
    for page_data in structured_pages:
        page_num = page_data['page_number']
        full_text = page_data['full_text']
        if not full_text.strip():
            continue
        # Use LLM to split the page text into logical chunks
        prompt = f"""
Analyze the following text from a document page and split it into logical, coherent chunks. Each chunk should be a self-contained section (e.g., a paragraph, heading with content, or related sentences). Keep chunks under {max_chunk_size} characters if possible, but prioritize logical breaks over strict size.

Text:
{full_text}

Output format: List each chunk on a new line, prefixed with "Chunk X: " where X is the number.
""".strip()
        messages = [{"role": "user", "content": prompt}]
        response = llm.chat(messages, max_tokens=1000)
        # Parse the response
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Chunk "):
                chunk_text = line.split(": ", 1)[1] if ": " in line else line
                if chunk_text.strip():
                    chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=chunk_text.strip()))
def split_pages_into_chunks(doc_id: str, pages: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[DocumentChunk]:
    """Legacy splitter for backward compatibility."""
    chunks: List[DocumentChunk] = []
    for i, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(DocumentChunk(doc_id=doc_id, page_number=i, text=chunk_text))
            if end == len(text):
                break
            start = max(end - chunk_overlap, start + 1)
    return chunks


def split_structured_pdf_into_chunks(doc_id: str, pdf_path: str, llm: IChatLLM, max_chunk_size: int = 1000) -> tuple[List[DocumentChunk], Dict[str, Any]]:
    """Intelligent LLM-based chunking: use LLM to split text into logical chunks."""
    pages_data, metadata = load_pdf_pages_structured(pdf_path)
    chunks: List[DocumentChunk] = []
    for page_data in pages_data:
        page_num = page_data['page_number']
        full_text = page_data['full_text']
        if not full_text.strip():
            continue
        # Use LLM to split the page text into logical chunks
        prompt = f"""
Analyze the following text from a document page and split it into logical, coherent chunks. Each chunk should be a self-contained section (e.g., a paragraph, heading with content, or related sentences). Keep chunks under {max_chunk_size} characters if possible, but prioritize logical breaks over strict size.

Text:
{full_text}

Output format: List each chunk on a new line, prefixed with "Chunk X: " where X is the number.
""".strip()
        messages = [{"role": "user", "content": prompt}]
        response = llm.chat(messages, max_tokens=1000)
        # Parse the response
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Chunk "):
                chunk_text = line.split(": ", 1)[1] if ": " in line else line
                if chunk_text.strip():
                    chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=chunk_text.strip()))
    return chunks, metadata

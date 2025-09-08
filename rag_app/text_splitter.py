from __future__ import annotations

from typing import List, Dict, Any
from .interfaces import DocumentChunk
from .pdf_parser import parse_pdf_advanced
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


def split_structured_pdf_into_chunks(doc_id: str, pdf_path: str, llm: IChatLLM, max_chunk_size: int = 1000) -> tuple[List[DocumentChunk], Dict[str, Any]]:
    """Intelligent LLM-based chunking: use LLM to split text into logical chunks while preserving structure."""
    pages_data, metadata = parse_pdf_advanced(pdf_path)
    chunks: List[DocumentChunk] = []

    # Store headings in metadata for structural indexing
    headings_map = {}
    for page_data in pages_data:
        page_num = page_data['page_number']
        headings = page_data.get('headings', [])
        if headings:
            headings_map[page_num] = headings

    metadata['headings'] = headings_map

    for page_data in pages_data:
        page_num = page_data['page_number']
        full_text = page_data['full_text']
        headings = page_data.get('headings') or []

        if not full_text.strip():
            continue

        # If we have detected headings, use them to guide chunking
        if headings:
            # Split text by headings to create structured chunks
            text_parts = []
            remaining_text = full_text

            for heading in headings:
                # Find the heading in the text
                heading_pos = remaining_text.find(heading)
                if heading_pos != -1:
                    # Add text before heading as a chunk (if not empty)
                    before_text = remaining_text[:heading_pos].strip()
                    if before_text:
                        text_parts.append(before_text)

                    # Find the end of this section (next heading or end of page)
                    remaining_text = remaining_text[heading_pos + len(heading):]

                    # Look for next heading
                    next_heading_pos = -1
                    for next_heading in headings[headings.index(heading) + 1:]:
                        pos = remaining_text.find(next_heading)
                        if pos != -1:
                            next_heading_pos = pos
                            break

                    if next_heading_pos != -1:
                        section_text = remaining_text[:next_heading_pos].strip()
                        remaining_text = remaining_text[next_heading_pos:]
                    else:
                        section_text = remaining_text.strip()
                        remaining_text = ""

                    # Create chunk for this section
                    if section_text:
                        chunk_text = f"{heading}\n{section_text}"
                        if len(chunk_text) <= max_chunk_size:
                            chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=chunk_text, section=heading))
                        else:
                            # If too long, split further
                            sub_chunks = _split_long_text(chunk_text, max_chunk_size)
                            for sub_chunk in sub_chunks:
                                chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=sub_chunk, section=heading))

            # Add any remaining text
            if remaining_text.strip():
                text_parts.append(remaining_text.strip())

            # Process text parts that weren't associated with headings
            for part in text_parts:
                if len(part) <= max_chunk_size:
                    chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=part))
                else:
                    sub_chunks = _split_long_text(part, max_chunk_size)
                    for sub_chunk in sub_chunks:
                        chunks.append(DocumentChunk(doc_id=doc_id, page_number=page_num, text=sub_chunk))
        else:
            # Fallback to LLM-based chunking if no headings detected
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


def _split_long_text(text: str, max_size: int) -> List[str]:
    """Split long text into smaller chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

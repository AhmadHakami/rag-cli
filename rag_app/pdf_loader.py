from __future__ import annotations

from typing import List, Dict, Any
import pdfplumber


def load_pdf_pages_structured(path: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load PDF pages with structured text: paragraphs, headings, etc., and metadata."""
    pages_data: List[Dict[str, Any]] = []
    with pdfplumber.open(path) as pdf:
        # Extract metadata
        metadata = pdf.metadata or {}
        title = metadata.get('Title', 'Unknown')
        author = metadata.get('Author', 'Unknown')
        subject = metadata.get('Subject', 'Unknown')
        
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Extract characters with font info for heading detection
            chars = page.chars
            # Simple heuristic: lines with larger font size or bold are headings
            lines = []
            current_line = []
            current_font_size = None
            for char in chars:
                if char['text'] == '\n':
                    if current_line:
                        line_text = ''.join(c['text'] for c in current_line)
                        font_sizes = [c['size'] for c in current_line]
                        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                        is_heading = avg_font_size > 14  # heuristic
                        lines.append({
                            'text': line_text.strip(),
                            'font_size': avg_font_size,
                            'is_heading': is_heading,
                        })
                        current_line = []
                else:
                    current_line.append(char)
            if current_line:
                line_text = ''.join(c['text'] for c in current_line)
                font_sizes = [c['size'] for c in current_line]
                avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                is_heading = avg_font_size > 14
                lines.append({
                    'text': line_text.strip(),
                    'font_size': avg_font_size,
                    'is_heading': is_heading,
                })
            pages_data.append({
                'page_number': i,
                'full_text': text.strip(),
                'lines': lines,
            })
    return pages_data, metadata


def load_pdf_pages(path: str) -> List[str]:
    """Legacy function for backward compatibility."""
    structured = load_pdf_pages_structured(path)
    return [p['full_text'] for p in structured]

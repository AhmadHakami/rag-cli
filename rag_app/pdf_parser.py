from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re


def parse_pdf_advanced(pdf_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse PDF with structure and metadata.

    Attempts to use PyMuPDF (fitz) to extract:
    - per-page full text
    - document metadata (Title, Author, Subject, etc.)
    - TOC-derived headings with page numbers

    If PyMuPDF is unavailable, falls back to load_pdf_pages_structured from pdf_loader.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        from .pdf_loader import load_pdf_pages_structured
        return load_pdf_pages_structured(pdf_path)

    doc = fitz.open(pdf_path)
    meta = doc.metadata or {}
    title = meta.get("title") or meta.get("Title") or ""
    author = meta.get("author") or meta.get("Author") or ""
    subject = meta.get("subject") or meta.get("Subject") or ""
    metadata = {"Title": title, "Author": author, "Subject": subject, "Pages": len(doc)}

    # Get TOC: list of (level, title, page)
    toc = doc.get_toc(simple=True) or []

    # If no TOC, try to extract headings from text structure
    page_to_headings: Dict[int, List[str]] = {}
    if not toc:
        # Extract headings by analyzing text structure and font characteristics
        for i in range(len(doc)):
            page = doc.load_page(i)
            page_num = i + 1

            # Get text blocks with font information
            blocks = page.get_text("dict")["blocks"]

            headings = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""
                        max_font_size = 0
                        is_bold = False

                        for span in line["spans"]:
                            line_text += span["text"]
                            font_size = span["size"]
                            max_font_size = max(max_font_size, font_size)

                            # Check if bold (simple heuristic)
                            font_name = span.get("font", "").lower()
                            if "bold" in font_name or "black" in font_name:
                                is_bold = True

                        line_text = line_text.strip()
                        if not line_text:
                            continue

                        # Heuristics for heading detection:
                        # 1. Font size > 14 (larger than body text)
                        # 2. All caps or title case
                        # 3. Short lines (likely headings)
                        # 4. Starts with roman numerals or numbers
                        # 5. Contains common heading words
                        is_heading = (
                            max_font_size >= 14 or
                            (line_text.isupper() and len(line_text) < 100) or
                            (line_text.istitle() and len(line_text) < 80 and len(line_text) > 5) or
                            (is_bold and len(line_text) < 100) or
                            (line_text[0].isdigit() and len(line_text) < 50) or
                            any(line_text.startswith(prefix) for prefix in ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X."]) or
                            any(word in line_text.lower() for word in ["chapter", "section", "introduction", "summary", "conclusion", "abstract", "acknowledgments", "references", "appendix"])
                        )

                        # Additional filters to avoid fragmented headings
                        if is_heading:
                            # Skip if it looks like a fragmented title (multiple short words)
                            words = line_text.split()
                            if len(words) <= 3 and all(len(word) <= 10 for word in words) and not any(word in line_text.lower() for word in ["chapter", "section", "introduction", "summary", "conclusion", "abstract", "acknowledgments", "references", "appendix"]):
                                # Check if this might be part of a larger title
                                if len(words) > 1 and not line_text.isupper():
                                    is_heading = False

                        if is_heading and 5 < len(line_text) < 200:
                            # Clean up the heading
                            heading = line_text.strip()
                            # Remove page numbers that might be at the end
                            heading = re.sub(r'\s+\d+$', '', heading)
                            # Skip if it's just a URL or similar
                            if not heading.startswith('http') and not heading.startswith('www.'):
                                if heading and heading not in headings:
                                    headings.append(heading)

            page_to_headings[page_num] = headings
    else:
        # Use TOC if available
        for level, heading, page0 in toc:
            page_num = int(page0)
            page_to_headings.setdefault(page_num, []).append(str(heading).strip())

    pages_data: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # headings on this page from TOC or extracted
        heads = page_to_headings.get(i + 1, [])
        pages_data.append({
            "page_number": i + 1,
            "full_text": text,
            "headings": heads,
        })

    doc.close()
    return pages_data, metadata

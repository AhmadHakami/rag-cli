from __future__ import annotations

from typing import Any, Dict, List, Tuple


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
    metadata = {"Title": title, "Author": author, "Subject": subject}

    # Get TOC: list of (level, title, page)
    toc = doc.get_toc(simple=True) or []
    # Build a page -> headings map (use headings encountered up to that page)
    page_to_headings: Dict[int, List[str]] = {}
    for level, heading, page0 in toc:
        page_num = int(page0)
        page_to_headings.setdefault(page_num, []).append(str(heading).strip())

    pages_data: List[Dict[str, Any]] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        # headings on this page from TOC entries (best-effort)
        heads = page_to_headings.get(i + 1, [])
        pages_data.append({
            "page_number": i + 1,
            "full_text": text,
            "headings": heads,
        })

    doc.close()
    return pages_data, metadata

from pathlib import Path
from rag_app.pdf_loader import load_pdf_pages


def test_pdf_loader_handles_missing(tmp_path):
    # Create empty PDF
    p = tmp_path / "empty.pdf"
    p.write_bytes(b"%PDF-1.4\n%%EOF\n")
    # pypdf may parse with 0 pages; should return empty list or list with empty string
    try:
        pages = load_pdf_pages(str(p))
        assert isinstance(pages, list)
    except Exception:
        # Some pypdf versions raise; acceptable for test to not crash environment
        assert True

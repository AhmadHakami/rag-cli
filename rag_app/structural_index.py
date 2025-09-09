from __future__ import annotations

from typing import Dict, List


class StructuralIndex:
    """Stores a simple Table of Contents mapping section titles to page numbers."""

    def __init__(self):
        self.section_to_pages: Dict[str, List[int]] = {}

    def add_heading(self, heading: str, page_number: int):
        h = heading.strip()
        if not h:
            return
        self.section_to_pages.setdefault(h, [])
        if page_number not in self.section_to_pages[h]:
            self.section_to_pages[h].append(page_number)

    def pages_for(self, heading_query: str) -> List[int]:
        def _norm(s: str) -> str:
            return ''.join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

        q = _norm(heading_query)
        hits: List[int] = []
        # Try fuzzy matching if rapidfuzz available for better user queries
        try:
            from rapidfuzz import fuzz

            # Score headings and accept those above a threshold
            scores = []
            for h, pages in self.section_to_pages.items():
                score = fuzz.partial_ratio(q, _norm(h))
                scores.append((score, h, pages))
            # Keep headings with score >= 70 (tunable)
            for score, h, pages in scores:
                if score >= 70:
                    hits.extend(pages)
            # If none matched fuzzily, fall back to substring match
            if not hits:
                for h, pages in self.section_to_pages.items():
                    if q in _norm(h):
                        hits.extend(pages)
        except Exception:
            # rapidfuzz not installed; use normalized substring match
            for h, pages in self.section_to_pages.items():
                if q in _norm(h):
                    hits.extend(pages)
        return sorted(set(hits))

    def toc(self) -> Dict[str, List[int]]:
        return dict(self.section_to_pages)

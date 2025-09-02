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
        q = heading_query.strip().lower()
        hits: List[int] = []
        for h, pages in self.section_to_pages.items():
            if q in h.lower():
                hits.extend(pages)
        return sorted(set(hits))

    def toc(self) -> Dict[str, List[int]]:
        return dict(self.section_to_pages)

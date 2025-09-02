from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re
from .interfaces import DocumentChunk


class LexicalIndex:
    """A simple inverted index for keyword/phrase lookups and term frequencies."""

    def __init__(self):
        self.term_to_occurrences: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        # Use Counter as default factory to avoid lambda (pickle-friendly)
        self.page_counts: Dict[int, Counter] = defaultdict(Counter)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\w']+", text.lower())

    def add_chunks(self, chunks: List[DocumentChunk]):
        for idx, ch in enumerate(chunks):
            tokens = self._tokenize(ch.text)
            for t in tokens:
                self.term_to_occurrences[t].append((ch.page_number, idx))
                self.page_counts[ch.page_number][t] += 1

    def term_frequency(self, term: str) -> int:
        return len(self.term_to_occurrences.get(term.lower(), []))

    def term_frequency_by_page(self, term: str) -> Dict[int, int]:
        term = term.lower()
        result: Dict[int, int] = {}
        for page, counts in self.page_counts.items():
            if term in counts:
                result[page] = counts[term]
        return result

    def find_pages_with_term(self, term: str) -> List[int]:
        return sorted({p for p, _ in self.term_to_occurrences.get(term.lower(), [])})

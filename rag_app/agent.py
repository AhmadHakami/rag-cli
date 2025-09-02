from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Sequence
import numpy as np
from tqdm import tqdm

from .interfaces import (
    IEmbedder,
    IChatLLM,
    IVectorStore,
    IQueryRewriter,
    IAnswerGenerator,
    DocumentChunk,
    RetrievalResult,
)
from .text_splitter import split_structured_pdf_into_chunks
from .pdf_loader import load_pdf_pages
from .llm import relevance_yes_no
from .lexical_index import LexicalIndex
from .structural_index import StructuralIndex
from . import cache as cache_util


@dataclass
class AgentProgress:
    steps: List[str]


class AgenticRAG:
    def __init__(
        self,
        embedder: IEmbedder,
        llm: IChatLLM,
        vector_store: IVectorStore,
        rewriter: IQueryRewriter,
        answerer: IAnswerGenerator,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.llm = llm
        self.store = vector_store
        self.rewriter = rewriter
        self.answerer = answerer
        self.top_k = top_k
        self._history = []
        # Additional indexes
        self.lex_index = LexicalIndex()
        self.struct_index = StructuralIndex()

    def ingest_pdf(self, pdf_path: str, doc_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
        steps = ["hash", "check_cache", "load_or_parse", "embed", "index", "save_cache"]
        with tqdm(total=len(steps), desc="Ingestion", unit="step") as pbar:
            # Step 1: hash
            pbar.set_description("Ingestion: hashing PDF")
            pdf_hash = cache_util.compute_pdf_hash(pdf_path)
            pbar.set_postfix_str("hashed")
            pbar.update(1)

            # Step 2: cache key
            pbar.set_description("Ingestion: checking cache")
            embedder_fp = f"{self.embedder.__class__.__name__}:{getattr(self.embedder, 'model', '')}"
            key = cache_util.make_cache_key(pdf_hash, doc_id, chunk_size, embedder_fp)
            cached = cache_util.has_cache(key)
            pbar.set_postfix(cache={"hit": bool(cached)})
            pbar.update(1)

            chunks: List[DocumentChunk] = []
            metadata = {}
            if cached:
                pbar.set_description("Ingestion: loading from cache")
                try:
                    data = cache_util.load_cache(key)
                    # restore store and indexes
                    self.store.embeddings = data["embeddings"]
                    self.store.metadatas = data["chunks"]
                    self.store.metadata = data.get("metadata", {})
                    self.lex_index = data["lex_index"]
                    self.struct_index = data["struct_index"]
                    chunks = list(self.store.metadatas)
                    pbar.set_postfix(loaded=len(chunks))
                    pbar.update(1)  # load_or_parse
                    # skip embed/index/save since already present
                    pbar.update(3)
                    return len(chunks)
                except Exception:
                    # Corrupt cache; invalidate and rebuild
                    cache_util.invalidate_cache(key)
                    pbar.set_postfix_str("cache invalid, rebuilding")

            # Step 3: parse + chunk with LLM
            pbar.set_description("Ingestion: parsing & chunking with LLM")
            chunks, metadata = split_structured_pdf_into_chunks(doc_id, pdf_path, self.llm, max_chunk_size=chunk_size)
            if not chunks:
                pbar.set_postfix_str("no chunks")
                pbar.update(1)  # load_or_parse
                pbar.update(3)  # embed/index/save
                return 0
            pbar.set_postfix(chunks=len(chunks))
            pbar.update(1)  # load_or_parse

            # Step 4: embed
            pbar.set_description("Ingestion: embedding chunks")
            embeddings = self.embedder.embed_texts([c.text for c in chunks])
            self.store.add(embeddings, chunks)
            self.store.metadata = metadata
            pbar.set_postfix(dim=int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0, n=len(chunks))
            pbar.update(1)

            # Step 5: index (lexical/structural)
            pbar.set_description("Ingestion: building indexes")
            self.lex_index.add_chunks(chunks)
            for c in chunks:
                first_line = c.text.strip().split("\n")[0]
                if 0 < len(first_line) <= 80:
                    if first_line.endswith(":") or first_line.istitle():
                        self.struct_index.add_heading(first_line.rstrip(":"), c.page_number)
            pbar.set_postfix(indexed_pages=len({c.page_number for c in chunks}))
            pbar.update(1)

            # Step 6: save cache
            pbar.set_description("Ingestion: saving cache")
            payload = {
                "embeddings": self.store.embeddings,
                "chunks": self.store.metadatas,
                "metadata": self.store.metadata,
                "lex_index": self.lex_index,
                "struct_index": self.struct_index,
            }
            cache_util.save_cache(key, payload)
            pbar.set_postfix_str("saved")
            pbar.update(1)
            return len(chunks)

    def ask(self, question: str) -> str:
        # Progress across: rewrite -> embed query -> retrieve -> grade -> answer
        steps = ["rewrite", "route", "retrieve", "grade", "answer"]
        with tqdm(total=len(steps), desc="Query", unit="step") as pbar:
            # 1) rewrite
            pbar.set_description("Query: rewriting question")
            rewritten = self.rewriter.rewrite(question, self._history)
            pbar.set_postfix_str("rewritten")
            pbar.update(1)
            # 2) route
            pbar.set_description("Query: routing")
            route = self._route_query(rewritten)
            pbar.set_postfix(route=route)
            pbar.update(1)
            if route == "lexical":
                ans = self._answer_lexical(rewritten)
                self._history.append((question, ans))
                return ans
            if route == "structural":
                ans = self._answer_structural(rewritten)
                self._history.append((question, ans))
                return ans
            # 3) embed query
            pbar.set_description("Query: embedding question")
            q_emb = self.embedder.embed_texts([rewritten])[0]
            pbar.set_postfix_str("embedded")
            pbar.update(1)
            # 4) retrieve
            pbar.set_description("Query: retrieving")
            retrieved: List[RetrievalResult] = self.store.search(q_emb, top_k=self.top_k)
            pbar.set_postfix(found=len(retrieved))
            pbar.update(1)
            # 5) grade relevance strictly, but if none relevant, include top retrieved
            pbar.set_description("Query: grading relevance")
            relevant: List[DocumentChunk] = []
            for r in retrieved:
                if relevance_yes_no(self.llm, rewritten, r.chunk):
                    relevant.append(r.chunk)
            if not relevant:
                # If no chunks graded relevant, include top 3 retrieved to avoid empty context
                relevant = [r.chunk for r in retrieved[:3]]
            pbar.set_postfix(relevant=len(relevant))
            pbar.update(1)
            # 6) answer
            pbar.set_description("Query: generating answer")
            answer = self.answerer.answer(rewritten, relevant)
            pbar.set_postfix_str("done")
            pbar.update(1)

        self._history.append((question, answer))
        return answer

    # --- Routing and direct answerers ---
    def _route_query(self, q: str) -> str:
        ql = q.lower()
        if any(k in ql for k in ["how many times", "frequency of", "count of", "how often", "how many times is", "how many times was", "word count"]):
            return "lexical"
        if any(k in ql for k in ["table of contents", "toc", "section", "page number of", "which page", "pages for", "page of"]):
            return "structural"
        return "semantic"

    def _answer_lexical(self, q: str) -> str:
        import re
        # try to extract the quoted term first; otherwise last word
        m = re.search(r"'([^']+)'|\"([^\"]+)\"", q)
        if m:
            term = (m.group(1) or m.group(2)).strip()
        else:
            words = re.findall(r"[\w']+", q.lower())
            term = words[-1] if words else ""
        if not term:
            return "I don't know based on the document."
        per_page = self.lex_index.term_frequency_by_page(term)
        total = sum(per_page.values())
        if total == 0:
            return "I don't know based on the document."
        parts = [f"p.{p}: {c}" for p, c in sorted(per_page.items())]
        return f"'{term}' appears {total} times across pages: " + ", ".join(parts)

    def _answer_structural(self, q: str) -> str:
        ql = q.lower()
        if (
            "table of contents" in ql
            or ql.strip() == "toc"
            or "list all sections" in ql
            or ("what" in ql and "sections" in ql)
            or ("sections" in ql and ("list" in ql or "show" in ql))
        ):
            toc = self.struct_index.toc()
            if not toc:
                return "I don't know based on the document."
            items = [f"{k}: pages {', '.join(str(p) for p in v)}" for k, v in toc.items()]
            return "Table of Contents:\n" + "\n".join(items)
        import re
        m = re.search(r"page number of\s+(.+)$", ql)
        if m:
            topic = m.group(1).strip()
            pages = self.struct_index.pages_for(topic)
            if not pages:
                return "I don't know based on the document."
            return f"The page number(s) for '{topic}' are: {', '.join(str(p) for p in pages)}"
        m2 = re.search(r"section\s+(.+)$", ql)
        if m2:
            section = m2.group(1).strip()
            pages = self.struct_index.pages_for(section)
            if not pages:
                return "I don't know based on the document."
            return f"Section '{section}' is on pages: {', '.join(str(p) for p in pages)}"
        return "I don't know based on the document."

    def summarize_document(self) -> str:
        """Generate a logical, deep summary of the entire document for understanding before Q&A."""
        all_texts = [chunk.text for chunk in self.store.metadatas]
        if not all_texts:
            return "No document content available to summarize."
        full_text = "\n".join(all_texts)
        # Extract metadata if available
        metadata = getattr(self.store, 'metadata', {})
        title = metadata.get('Title', 'Unknown')
        author = metadata.get('Author', 'Unknown')
        subject = metadata.get('Subject', 'Unknown')
        prompt = f"""
Provide a comprehensive, logical summary of the following document. Include:
- Document Type: Based on content, e.g., CV, Report, etc.
- Title: {title}
- Author: {author}
- Subject: {subject}
- Table of Contents: Extract or infer main sections/headings.
- General Information: Key facts, dates, etc.
- Structure the summary with sections for clarity.

Document Text:
{full_text[:8000]}  # Limit to avoid token overflow
""".strip()
        messages = [{"role": "user", "content": prompt}]
        return self.llm.chat(messages, max_tokens=1500)

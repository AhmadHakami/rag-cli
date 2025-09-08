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
from .messages import NO_ANSWER, no_answer_response


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

            # Build structural index from PDF headings stored in metadata
            if 'headings' in metadata:
                for page_num, headings in metadata['headings'].items():
                    for heading in headings:
                        self.struct_index.add_heading(heading, page_num)

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
        # Handle metadata questions directly
        ql = question.lower()
        if "title" in ql and ("document" in ql or "paper" in ql or "what is" in ql):
            title = self.store.metadata.get("Title", "")
            if title:
                return f"The title of the document is: {title}"
            else:
                return no_answer_response(question)
        elif ("author" in ql or "who" in ql) and ("wrote" in ql or "is" in ql):
            author = self.store.metadata.get("Author", "")
            if author:
                return f"The author of the document is: {author}"
            else:
                return no_answer_response(question)
        elif "how many" in ql and "pages" in ql:
            pages = self.store.metadata.get("Pages", 0)
            if pages:
                return f"The document has {pages} pages."
            else:
                return no_answer_response(question)

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
        if any(k in ql for k in [
            "how many times",
            "frequency of",
            "count of",
            "how often",
            "word count",
            "occurrence of",
            "mentioned",
        ]):
            return "lexical"
        if any(k in ql for k in [
            "table of contents",
            "toc",
            "section",
            "sections",
            "page number of",
            "which page",
            "pages for",
            "page of",
            "extract content",
            "content of page",
            "extract page",
            "get page",
            "show page",
            "read page",
            "related to",
            "page is",
        ]):
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
            return no_answer_response(q)
        per_page = self.lex_index.term_frequency_by_page(term)
        total = sum(per_page.values())
        if total == 0:
            return no_answer_response(q)
        parts = [f"p.{p}: {c}" for p, c in sorted(per_page.items())]
        # Lexical answers are direct counts from the keyword index; avoid LLM-style citations
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
                return no_answer_response(q)
            items = [f"{k}: pages {', '.join(str(p) for p in v)}" for k, v in toc.items()]
            return "Table of Contents:\n" + "\n".join(items)
        import re
        m = re.search(r"page number of\s+(.+)$", ql)
        if m:
            topic = m.group(1).strip()
            pages = self.struct_index.pages_for(topic)
            if not pages:
                return no_answer_response(q)
            # Structural answers reflect metadata/TOC; do not add (p.X) citations
            return f"The page number(s) for '{topic}' are: {', '.join(str(p) for p in pages)}"
        
        # Handle "which page" questions
        m_page = re.search(r"which page(?:s)?(?:\s+is|\s+are)?(?:\s+related to|\s+contains|\s+has|\s+for)?\s+(.+?)(?:\s+section|\s+part|\?|$)", ql)
        if m_page:
            topic = m_page.group(1).strip()
            # Clean up the topic
            topic = re.sub(r'^(the\s+)', '', topic)  # Remove leading "the"
            pages = self.struct_index.pages_for(topic)
            if not pages:
                return no_answer_response(q)
            return f"The {topic} section is on page(s): {', '.join(str(p) for p in pages)}"
        
        m2 = re.search(r"section\s+(.+)$", ql)
        if m2:
            section = m2.group(1).strip()
            pages = self.struct_index.pages_for(section)
            if not pages:
                return no_answer_response(q)
            return f"Section '{section}' is on pages: {', '.join(str(p) for p in pages)}"

        # Handle page content extraction
        m3 = re.search(r"(?:extract|get|show|read|what(?:'s| is))\s+(?:the\s+)?(?:content|text|summary)\s+(?:of\s+)?(?:page\s+)?(\d+)", ql)
        if m3:
            page_num = int(m3.group(1))
            # Find chunks from this page
            page_chunks = [c for c in self.store.metadatas if c.page_number == page_num]
            if not page_chunks:
                return f"I couldn't find content for page {page_num}. The document may have {self.store.metadata.get('Pages', 'unknown')} pages."
            # Combine all chunks from this page
            full_content = "\n\n".join(c.text for c in page_chunks)
            return f"Content of page {page_num}:\n\n{full_content}"

        # Handle section content extraction
        m4 = re.search(r"(?:extract|get|show|read|what(?:'s| is))\s+(?:the\s+)?(?:full\s+)?(?:content|text|summary)\s+(?:of\s+)?(?:the\s+)?(\w+)\s+section", ql)
        if m4:
            section = m4.group(1)
            pages = self.struct_index.pages_for(section)
            if not pages:
                return no_answer_response(q)
            # Get content from the first page of this section
            page_chunks = [c for c in self.store.metadatas if c.page_number == pages[0]]
            if not page_chunks:
                return f"I found the {section} section on page {pages[0]}, but couldn't retrieve the content."
            full_content = "\n\n".join(c.text for c in page_chunks)
            return f"Content of {section} section (page {pages[0]}):\n\n{full_content}"
        return no_answer_response(q)

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

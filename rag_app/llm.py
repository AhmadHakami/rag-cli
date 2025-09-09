from __future__ import annotations

from typing import List, Dict, Sequence, Tuple
import requests
from .config import SETTINGS
from .messages import NO_ANSWER, no_answer_response
from .interfaces import IChatLLM, IQueryRewriter, IAnswerGenerator, DocumentChunk


SYSTEM_REWRITER = """
You are a helpful assistant that rewrites follow-up questions into standalone questions using the chat history. Keep it brief and relevant to the document domain.
""".strip()

SYSTEM_GRADER = """
You are a strict relevance grader. Given a user question and a context chunk, respond with ONLY 'YES' if the chunk is relevant, otherwise 'NO'.
""".strip()

SYSTEM_ANSWER = """
You are a professional assistant that answers questions ONLY using the provided document context below. Do not use any external knowledge or assumptions.

If the answer is not present in the context, respond kindly and professionally with a brief, interactive clarification tailored to the user's question (e.g., ask for a page/section or keywords). Do not invent information.

Citation policy (mandatory):
- For any factual claim taken from a context chunk, add an inline citation of the form (p. N) where N is the exact page_number from the chunk.
- If a claim is supported by multiple chunks from different pages, include multiple citations like (p. 3; p. 5).
- Never invent page numbers. If the supporting chunk has no page information, do not add a page citation.
- For structural answers (TOC, page numbers for sections, metadata like Title/Author), you may return the result without (p.N) citations.

Also:
- At the end of the answer, include a short "Sources:" list with entries like "p. N: <first 80 chars of chunk>..." for each chunk used in the answer.
- Preserve the user's question and, when available, use the chat history to resolve follow-ups and coreferences.

Context:
{context}

Question: {question}
"""


class OpenAIChat(IChatLLM):
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None):
        self.model = model or SETTINGS.openai_model
        self.api_key = api_key or SETTINGS.openai_api_key
        self.base_url = (base_url or SETTINGS.openai_base_url or "https://api.openai.com/v1").rstrip("/")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            prefix = "System"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = "User"
        lines.append(f"{prefix}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


class OllamaChat(IChatLLM):
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or SETTINGS.ollama_model
        self.base_url = (base_url or SETTINGS.ollama_base_url).rstrip("/")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "options": {"temperature": temperature}}
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code == 404:
            # Fallback to /api/generate for older Ollama versions
            gen_url = f"{self.base_url}/api/generate"
            prompt = _messages_to_prompt(messages)
            gen_payload = {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
            gen_resp = requests.post(gen_url, json=gen_payload, timeout=300)
            gen_resp.raise_for_status()
            data = gen_resp.json()
            return (data.get("response") or data.get("text") or "").strip()
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


class MockChat(IChatLLM):
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        # Very small rule-based output for tests
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if "relevant?" in last_user.lower():
            return "YES"
        if messages and "rewrite" in (messages[0].get("content", "").lower()):
            return last_user
        return no_answer_response(last_user)


class FallbackChat(IChatLLM):
    """Try multiple chat backends in order until one succeeds."""

    def __init__(self, llms: List[IChatLLM]):
        if not llms:
            raise ValueError("FallbackChat requires at least one LLM")
        self._llms = llms

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int | None = None) -> str:
        last_err: Exception | None = None
        for m in self._llms:
            try:
                return m.chat(messages, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        raise RuntimeError(f"All chat backends failed: {last_err}")


class QueryRewriter(IQueryRewriter):
    def __init__(self, llm: IChatLLM):
        self.llm = llm

    def rewrite(self, question: str, chat_history: Sequence[Tuple[str, str]] | None = None) -> str:
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in (chat_history or [])])
        messages = [
            {"role": "system", "content": SYSTEM_REWRITER},
            {"role": "user", "content": f"Chat History:\n{history_text}\nQuestion: {question}\nRewrite into a standalone question."},
        ]
        return self.llm.chat(messages)


class AnswerGenerator(IAnswerGenerator):
    def __init__(self, llm: IChatLLM):
        self.llm = llm

    def answer(self, question: str, contexts: Sequence[DocumentChunk], chat_history: Sequence[Tuple[str, str]] | None = None) -> str:
        joined = []
        for c in contexts:
            # include section if present for better grounding
            sec = f"[{c.section}] " if getattr(c, 'section', None) else ""
            joined.append(f"[p.{c.page_number}] {sec}{c.text}")
        context_text = "\n---\n".join(joined)
        # If chat_history exists, include as system context to help with follow-ups
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in (chat_history or [])])
        system_content = SYSTEM_ANSWER.format(context=context_text, question=question)
        if history_text:
            system_content = f"Previous interaction history:\n{history_text}\n\n" + system_content
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Please answer using the context above. Use inline citations and list sources at the end."},
        ]
        out = self.llm.chat(messages)

        # Post-process: if the model didn't include any (p. N) citations, try to inject them deterministically
        import re

        def _token_set(text: str) -> set:
            return set(re.findall(r"\w{3,}", text.lower()))

        def _inject_citations(answer_text: str, contexts: Sequence[DocumentChunk]) -> str:
            # Build token sets for contexts
            ctx_tokens = {}
            for c in contexts:
                ctx_tokens.setdefault(c.page_number, set()).update(_token_set(c.text))

            # Split answer into sentences
            sents = re.split(r'(?<=[.!?])\s+', answer_text)
            annotated = []
            for s in sents:
                if not s.strip():
                    continue
                stokens = _token_set(s)
                # score pages by overlap
                scores = []
                for pnum, toks in ctx_tokens.items():
                    overlap = len(stokens & toks)
                    scores.append((overlap, pnum))
                scores.sort(reverse=True)
                if scores and scores[0][0] >= 3:
                    # attach top page as citation
                    top_pages = [str(p) for sc, p in scores if sc >= 3]
                    citation = f" (p. {'; p. '.join(top_pages)})"
                    annotated.append(s + citation)
                else:
                    annotated.append(s)
            return " ".join(annotated)

        if contexts and not re.search(r"\(p\.\s*\d+\)", out):
            injected = _inject_citations(out, contexts)
            # Append concise sources list
            seen = []
            srcs = []
            for c in contexts:
                if c.page_number not in seen:
                    seen.append(c.page_number)
                    snippet = (c.text.strip()[:80].replace('\n', ' ') + '...') if c.text else ''
                    srcs.append(f"p. {c.page_number}: {snippet}")
            out = injected.strip() + "\n\nSources:\n" + "\n".join(srcs)
        return out


def relevance_yes_no(llm: IChatLLM, question: str, chunk: DocumentChunk) -> bool:
    messages = [
        {"role": "system", "content": SYSTEM_GRADER},
        {"role": "user", "content": f"Question: {question}\nContext: {chunk.text}\nRelevant?"},
    ]
    out = llm.chat(messages).strip().upper()
    return out.startswith("Y")

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

If the answer is not present in the context, respond kindly and professionally with a brief, interactive clarification tailored to the user's question (e.g., ask for a page/section or keywords). Do not include citations in this case, and vary the wording naturally.

Citation policy:
- Include page numbers in parentheses like (p. 12) ONLY when citing specific facts that are explicitly present in the provided context snippets.
- Never invent or infer page numbers.
- Do NOT include citations for metadata-only facts or structural/TOC information; cite only text-derived content present in the snippets.

Explain connections and suggest follow-up questions based on the context.

Context:
{context}

Question: {question}
""".strip()


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

    def answer(self, question: str, contexts: Sequence[DocumentChunk]) -> str:
        joined = []
        for c in contexts:
            joined.append(f"[p.{c.page_number}] {c.text}")
        context_text = "\n---\n".join(joined)
        messages = [
            {"role": "user", "content": SYSTEM_ANSWER.format(context=context_text, question=question)},
        ]
        return self.llm.chat(messages)


def relevance_yes_no(llm: IChatLLM, question: str, chunk: DocumentChunk) -> bool:
    messages = [
        {"role": "system", "content": SYSTEM_GRADER},
        {"role": "user", "content": f"Question: {question}\nContext: {chunk.text}\nRelevant?"},
    ]
    out = llm.chat(messages).strip().upper()
    return out.startswith("Y")

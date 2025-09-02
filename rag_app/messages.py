from __future__ import annotations

import re
from typing import List

# Backward-compatible constant (kept for imports), but code should prefer no_answer_response()
NO_ANSWER = "I couldn’t find that information in the document."


def _extract_keywords(question: str, max_terms: int = 3) -> List[str]:
	text = question.lower()
	tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text)
	stop = set(
		"the a an and or but if then else for with without from to of in on at by is are was were be been being this that these those which who whom whose what when where why how can could would should do does did as about into over under between among across per each such not no yes more most less least many few several any some every".split()
	)
	terms = [t for t in tokens if t not in stop]
	seen = set()
	uniq = []
	for t in terms:
		if t not in seen:
			seen.add(t)
			uniq.append(t)
		if len(uniq) >= max_terms:
			break
	return uniq


def no_answer_response(question: str) -> str:
	"""Generate a kind, professional, and interactive response when the answer isn't in the document.

	Tailors phrasing and suggestions to the user's question while avoiding citations or page numbers.
	"""
	kws = _extract_keywords(question, max_terms=2)
	hint = (
		f" You could try specifying terms like: {', '.join(kws)}." if kws else ""
	)
	variants = [
		"I couldn’t find that information in the document I have. Could you clarify what you’re looking for or share a page range?" + hint,
		"I didn’t see that detail in the provided pages. If you can rephrase or add a keyword or section name, I’ll check again." + hint,
		"That doesn’t appear in the text I can reference. Would you like me to look for related terms or a specific section?" + hint,
		"I wasn’t able to locate an answer in the document. A narrower question or a page/section hint would help me dig deeper." + hint,
	]
	# Pick a variant based on the question content for gentle diversity
	idx = (sum(ord(c) for c in question) or 0) % len(variants)
	return variants[idx]

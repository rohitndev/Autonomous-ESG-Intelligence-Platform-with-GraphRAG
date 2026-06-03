"""ESG sentence classifier.

The full platform uses HuggingFace **FinBERT** to label each sentence as
Environmental / Social / Governance with a confidence score. To keep the prototype
dependency-light and instantly runnable, this module reproduces the same *interface*
with a transparent rule-based (keyword-weighted) classifier. Swapping in FinBERT later
only means replacing :func:`classify_sentence` — every caller stays the same.
"""
from __future__ import annotations

import re
from collections import Counter

# Keyword lexicons per ESG pillar (a lightweight FinBERT proxy).
LEXICON: dict[str, list[str]] = {
    "Environmental": [
        "emission", "emissions", "nox", "carbon", "climate", "scope", "pollution",
        "water", "deforestation", "fossil", "renewable", "solar", "wind", "tailings",
        "waste", "net-zero", "taxonomy",
    ],
    "Social": [
        "labor", "labour", "overtime", "rights", "conflict", "mineral", "minerals",
        "community", "human", "safety", "health", "worker", "artisanal", "diversity",
    ],
    "Governance": [
        "board", "governance", "antitrust", "anti-competition", "disclosure", "audit",
        "due-diligence", "compliance", "oversight", "committee", "bribery", "csrd",
        "regulator", "reform",
    ],
}

_WORD_RE = re.compile(r"[a-zA-Z\-]+")


def classify_sentence(sentence: str) -> dict:
    """Return the dominant ESG pillar for a sentence with a pseudo-confidence.

    Confidence = winning pillar hits / total pillar hits. Falls back to ``Neutral``
    when no ESG keyword is present.
    """
    tokens = [t.lower() for t in _WORD_RE.findall(sentence)]
    scores: Counter[str] = Counter()
    for pillar, words in LEXICON.items():
        scores[pillar] = sum(tokens.count(w) for w in words)

    total = sum(scores.values())
    if total == 0:
        return {"label": "Neutral", "confidence": 0.0, "scores": dict(scores)}

    label, hits = scores.most_common(1)[0]
    return {
        "label": label,
        "confidence": round(hits / total, 3),
        "scores": dict(scores),
    }


def classify_document(text: str) -> list[dict]:
    """Split a document into sentences and classify each (FinBERT pipeline proxy)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    out = []
    for sent in sentences:
        result = classify_sentence(sent)
        result["sentence"] = sent
        out.append(result)
    return out

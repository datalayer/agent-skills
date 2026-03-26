# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Text Summarizer Skill.

Provides extractive text summarization using sentence scoring.
This module can be used both as a standalone skill (via SKILL.md discovery)
and as a package-based skill (via package + method reference).

Usage:
    from agent_skills.skills.text_summarizer import summarize_text
    result = summarize_text("Long text...", max_sentences=3)
"""

from __future__ import annotations

import re
from collections import Counter


def summarize_text(
    text: str,
    max_sentences: int = 5,
    method: str = "extractive",
) -> str:
    """Summarize text content.

    Args:
        text (str): The text to summarize.
        max_sentences (int): Maximum number of sentences in the summary.
        method (str): Summarization method - 'extractive' or 'key_points'.

    Returns:
        A summarized version of the input text.
    """
    if not text or not text.strip():
        return "No text provided to summarize."

    sentences = _split_sentences(text)

    if len(sentences) <= max_sentences:
        return text.strip()

    scored = _score_sentences(sentences, text)

    top = sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]
    # Preserve original order
    top_ordered = sorted(top, key=lambda x: x[2])

    if method == "key_points":
        return "\n".join(f"- {s[0].strip()}" for s in top_ordered)

    return " ".join(s[0].strip() for s in top_ordered)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def _score_sentences(
    sentences: list[str], full_text: str
) -> list[tuple[str, float, int]]:
    """Score sentences by word frequency relevance.

    Returns list of (sentence, score, original_index).
    """
    words = re.findall(r'\b\w+\b', full_text.lower())
    # Filter out very common English stop words
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'shall',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'and',
        'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
        'than', 'too', 'very', 'just', 'because', 'if', 'when', 'while',
        'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she',
        'they', 'them', 'their', 'we', 'you', 'i', 'me', 'my', 'your',
        'his', 'her', 'our', 'us',
    }
    filtered = [w for w in words if w not in stop_words and len(w) > 1]
    freq = Counter(filtered)

    scored = []
    for idx, sentence in enumerate(sentences):
        s_words = re.findall(r'\b\w+\b', sentence.lower())
        score = sum(freq.get(w, 0) for w in s_words if w not in stop_words)
        # Normalize by sentence length to avoid bias toward long sentences
        if s_words:
            score /= len(s_words)
        scored.append((sentence, score, idx))

    return scored

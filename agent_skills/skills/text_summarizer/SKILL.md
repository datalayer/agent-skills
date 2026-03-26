---
name: text-summarizer
description: Summarize text content using extractive and abstractive techniques. Use when the user asks for summaries, key points, or condensed versions of documents.
license: BSD-3-Clause
version: 1.0.0
tags:
  - nlp
  - summarization
  - text-processing
author: Datalayer
compatibility: Requires Python 3.10+
allowed-tools: Read
metadata:
  category: nlp
---

# Text Summarizer Skill

## Overview

Use this skill to produce concise summaries of text content. It supports both extractive summarization (selecting key sentences) and simple abstractive summarization (condensing content).

## When to Use

- The user asks for a summary, key points, or TL;DR of a document
- You need to condense long text before further processing
- You want to extract the most important sentences from a passage

## Quick Start

```python
from agent_skills.skills.text_summarizer import summarize_text

# Extractive summary (select top sentences)
result = summarize_text(text, max_sentences=5, method="extractive")

# Key points extraction
result = summarize_text(text, max_sentences=3, method="key_points")
```

## Available Scripts

### `summarize.py`

Summarize text from a file or stdin.

```bash
python scripts/summarize.py --file document.txt --max-sentences 5
python scripts/summarize.py --text "Long text here..." --method key_points
```

## Methods

| Method | Description |
|--------|-------------|
| `extractive` | Selects the most important sentences based on word frequency scoring |
| `key_points` | Extracts sentences as bullet-point key takeaways |

## Tips

- For very long documents, consider chunking the text first
- Extractive summarization preserves original wording
- Use `max_sentences` to control output length

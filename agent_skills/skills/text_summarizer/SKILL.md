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

## Environment

- None required.

## Script Inventory

### `scripts/summarize.py`

- Method: `summarize(text: str, max_sentences: int = 5, method: str = "extractive") -> str`
- Required CLI params (one of):
- `--file <path>`
- `--text <content>`
- Optional CLI params:
- `--max-sentences <int>`
- `--method <extractive|key_points>`

## Usage Examples

```bash
python agent_skills/skills/text_summarizer/scripts/summarize.py --file document.txt --max-sentences 5
python agent_skills/skills/text_summarizer/scripts/summarize.py --text "Long text here" --method key_points --max-sentences 3
```

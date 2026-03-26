#!/usr/bin/env python3
# Copyright (c) 2025-2026 Datalayer, Inc.
#
# BSD 3-Clause License

"""Summarize text from a file or direct input.

Usage: python summarize.py --file document.txt --max-sentences 5
       python summarize.py --text "Long text..." --method key_points

Environment:
    (No environment variables required)
"""

import argparse
import sys


def summarize(
    text: str,
    max_sentences: int = 5,
    method: str = "extractive",
) -> str:
    """Summarize the given text content.

    Args:
        text (str): The text to summarize.
        max_sentences (int): Maximum sentences in output.
        method (str): 'extractive' or 'key_points'.

    Returns:
        Summarized text string.
    """
    from agent_skills.skills.text_summarizer import summarize_text

    return summarize_text(text, max_sentences=max_sentences, method=method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize text content")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to text file to summarize")
    group.add_argument("--text", type=str, help="Direct text input to summarize")
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=5,
        help="Maximum number of sentences in summary",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["extractive", "key_points"],
        default="extractive",
        help="Summarization method",
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            content = f.read()
    else:
        content = args.text

    result = summarize(content, max_sentences=args.max_sentences, method=args.method)
    print(result)

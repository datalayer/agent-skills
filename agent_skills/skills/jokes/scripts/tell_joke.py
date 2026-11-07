# Copyright (c) 2025-2026 Datalayer, Inc.
# Distributed under the terms of the Modified BSD License.

import argparse
import json
import random
import sys

# One joke, at random or on a topic, from the skill's own short, work-safe list.

JOKES = [
    (
        "data",
        "Why did the data scientist break up with the spreadsheet? Too many cells, not enough commitment.",
    ),
    ("data", "A SQL query walks into a bar, walks up to two tables and asks: may I join you?"),
    ("data", "There are two kinds of people: those who can extrapolate from incomplete data."),
    (
        "agents",
        "Why did the agent refuse the task? It was out of its scope, and it had read the Rules.",
    ),
    (
        "agents",
        "Two agents meet in a scene. One says: you go first. The other says: after you. The referee calls a draw.",
    ),
    (
        "agents",
        "My agent asked for approval before buying lunch. Now it asks before opening the fridge.",
    ),
    ("accounting", "Why did the accountant cross the road? To reconcile the other side."),
    ("accounting", "An accountant's favourite month? Close enough."),
    (
        "accounting",
        "Debits on the left, credits on the right, and the coffee in the middle: a balanced day.",
    ),
    ("computers", "There are 10 kinds of people: those who read binary, and those who don't."),
    ("computers", "I would tell you a UDP joke, but you might not get it."),
    ("computers", "Why do programmers prefer dark mode? Because light attracts bugs."),
]

TOPICS = sorted({topic for topic, _ in JOKES})


def tell(topic, count, rng=random):
    chosen = [joke for joke in JOKES if topic == "any" or joke[0] == topic]
    note = None
    if not chosen:
        chosen = list(JOKES)
        note = f"No joke about {topic!r} yet; one at random."
    picked = rng.sample(chosen, k=min(count, len(chosen)))
    return [{"topic": t, "joke": j} for t, j in picked], note


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One joke, at random or on a topic.")
    parser.add_argument("--topic", default="any")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    options = parser.parse_args()
    if not 1 <= options.count <= 5:
        print("Error: --count is 1 to 5", file=sys.stderr)
        sys.exit(2)
    jokes, note = tell(options.topic.strip().lower(), options.count)
    if options.format == "json":
        print(
            json.dumps(
                {"jokes": jokes, "topics": TOPICS, **({"note": note} if note else {})}, indent=2
            )
        )
    else:
        if note:
            print(note)
        print("\n".join(joke["joke"] for joke in jokes))

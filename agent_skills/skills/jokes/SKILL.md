---
name: jokes
description: A joke on request, from a small bundled list, by topic or at random. Use it when the person asks for a joke, a one-liner to open a message with, or something to lighten a reply — never in place of an answer they are waiting for.
license: BSD 3-Clause. LICENSE has complete terms
version: 1.0.0
tags:
  - fun
  - humor
  - demo
author: Datalayer
metadata:
  visibility: public
---

# Jokes Skill

One script, one joke. The jokes are the skill's own, short and work-safe,
tagged by topic; a topic nobody wrote a joke for gives a random one and says so.

## Required Environment Variables

- None required.

## Required Software

- None beyond Python.

## Invocation Contract

- Always call `run_skill_script` with:
  - `skill_name`: `jokes` (lowercase, exactly as returned by `list_skills`)
  - `script_name`: `tell_joke`
- Pass optional flags with `kwargs` using the documented keys below. Unknown keys exit with code `2`.

## Scripts API

### `script_name: tell_joke`

- Purpose: one joke, at random or on a topic.
- Positional `args`: none.
- Supported `kwargs`:
  - `topic`: `data | agents | accounting | computers | any` (default `any`)
  - `count`: how many, 1 to 5 (default `1`)
  - `format`: `text | json`
- Prints the joke(s); with `json`, each with its topic.

## Common Failure Modes

- Exit code `2`: an unknown `kwargs` key, or `count` outside 1 to 5.

---
name: crawl
description: Web crawling and extraction scripts for HTML pages, JS-rendered pages, links, tables, robots policy, and site crawling.
license: Proprietary. LICENSE.txt has complete terms
version: 1.0.0
tags:
  - web
  - scraping
  - crawling
  - http
author: Datalayer
---

# Crawl Skill

## Required Environment Variables

- None required.

## Invocation Contract

- Call `run_skill_script` with:
  - `skill_name`: `crawl`
  - `script_name`: one of `fetch_page`, `fetch_js_page`, `extract_links`, `extract_tables`, `crawl_site`, `check_robots`
- Put required positional parameters in `args`.
- Put optional value flags in `kwargs` using the keys listed below.
- For flag-only boolean options, use `args` (not `kwargs`).

## Scripts API

### `script_name: fetch_page`

- `args`: `["<url>"]`
- optional `kwargs`: `timeout`, `output`

### `script_name: fetch_js_page`

- `args`: `["<url>"]`
- optional `kwargs`: `wait`, `selector`, `timeout`, `output`
- optional boolean flag `--html`: pass via `args` as `"--html"`

### `script_name: extract_links`

- `args`: `["<url>"]`
- optional `kwargs`: `filter`, `timeout`, `output`
- `--absolute` is enabled by default.

### `script_name: extract_tables`

- `args`: `["<url>"]`
- optional `kwargs`: `format`, `timeout`, `output`
- `format`: `json | csv`

### `script_name: crawl_site`

- `args`: `["<start_url>"]`
- optional `kwargs`: `max_pages`, `max_depth`, `timeout`, `output`
- `--same-domain` is enabled by default.

### `script_name: check_robots`

- `args`: `["<url>"]`
- optional `kwargs`: `user_agent`

## `run_skill_script` Examples

- JS-rendered fetch with selector:

```json
{
  "skill_name": "crawl",
  "script_name": "fetch_js_page",
  "args": ["https://example.com"],
  "kwargs": {
    "wait": 1200,
    "selector": "main",
    "timeout": 45000
  }
}
```

- Include raw HTML output:

```json
{
  "skill_name": "crawl",
  "script_name": "fetch_js_page",
  "args": ["https://example.com", "--html"]
}
```

Notes:
- `fetch_js_page` requires Playwright + Chromium in the execution environment.

## Direct CLI Examples

```bash
python agent_skills/skills/crawl/scripts/fetch_page.py https://example.com --timeout 20
python agent_skills/skills/crawl/scripts/fetch_js_page.py https://example.com --wait 1200 --selector main
python agent_skills/skills/crawl/scripts/extract_links.py https://example.com --filter /docs
python agent_skills/skills/crawl/scripts/extract_tables.py https://example.com --format csv --output tables.csv
python agent_skills/skills/crawl/scripts/crawl_site.py https://example.com --max-pages 20 --max-depth 2 --output crawl.json
python agent_skills/skills/crawl/scripts/check_robots.py https://example.com --user-agent "DataBot/1.0"
```

---
name: crawl
description: Web crawling scripts for fetching pages, JS-rendered pages, links, tables, robots policy, and multi-page crawling. Use these scripts directly; provide URL, timeout, and optional output/filter params.
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

## Environment

- None required.

## Script Inventory

### `scripts/fetch_page.py`

- Method: `fetch_page(url: str, timeout: float = 30.0) -> dict`
- Required CLI params:
- positional `url`
- Optional CLI params:
- `--timeout/-t`
- `--output/-o`

### `scripts/fetch_js_page.py`

- Method: `fetch_js_page(url: str, wait_ms: int = 0, wait_selector: str | None = None, timeout: int = 30000) -> dict`
- Required CLI params:
- positional `url`
- Optional CLI params:
- `--wait/-w`
- `--selector/-s`
- `--timeout/-t`
- `--html`
- `--output/-o`
- Notes:
- Requires Playwright and Chromium.

### `scripts/extract_links.py`

- Method: `extract_links(url: str, absolute: bool = True, filter_pattern: str | None = None, timeout: float = 30.0) -> list[dict]`
- Required CLI params:
- positional `url`
- Optional CLI params:
- `--filter/-f`
- `--timeout/-t`
- `--output/-o`

### `scripts/extract_tables.py`

- Method: `extract_tables(url: str, timeout: float = 30.0) -> list[list[list[str]]]`
- Required CLI params:
- positional `url`
- Optional CLI params:
- `--format/-f` (`json|csv`)
- `--timeout/-t`
- `--output/-o`

### `scripts/crawl_site.py`

- Method: `crawl_site(start_url: str, max_pages: int = 10, max_depth: int = 2, same_domain_only: bool = True, timeout: float = 30.0) -> dict[str, dict]`
- Required CLI params:
- positional `url` (start URL)
- Optional CLI params:
- `--max-pages/-p`
- `--max-depth/-d`
- `--same-domain/-s`
- `--timeout/-t`
- `--output/-o`

### `scripts/check_robots.py`

- Method: `check_robots(url: str, user_agent: str = "*") -> dict`
- Required CLI params:
- positional `url`
- Optional CLI params:
- `--user-agent/-u`

## Usage Examples

```bash
python agent_skills/skills/crawl/scripts/fetch_page.py https://example.com --timeout 20
python agent_skills/skills/crawl/scripts/fetch_js_page.py https://example.com --wait 1200 --selector main
python agent_skills/skills/crawl/scripts/extract_links.py https://example.com --filter /docs
python agent_skills/skills/crawl/scripts/extract_tables.py https://example.com --format csv --output tables.csv
python agent_skills/skills/crawl/scripts/crawl_site.py https://example.com --max-pages 20 --max-depth 2 --output crawl.json
python agent_skills/skills/crawl/scripts/check_robots.py https://example.com --user-agent "DataBot/1.0"
```

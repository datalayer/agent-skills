---
name: github
description: GitHub API scripts for repo discovery, repo details, issue listing, and PR listing. Use scripts in scripts/ with required repo/query params and GITHUB_TOKEN.
license: Proprietary. LICENSE.txt has complete terms
version: 1.0.0
tags:
  - github
  - api
  - repositories
  - git
author: Datalayer
---

# GitHub Skill

## Environment

- `GITHUB_TOKEN` (required)

## Script Inventory

### `scripts/list_repos.py`

- Method: `list_repos(visibility="all", sort="updated", per_page=100) -> list[dict]`
- Required CLI params: none
- Optional CLI params:
- `--visibility` (`all|public|private`)
- `--sort` (`updated|created|pushed|full_name`)
- `--limit`
- `--format` (`table|json`)

### `scripts/get_repo.py`

- Method: `get_repo(owner: str, repo: str) -> dict`
- Required CLI params:
- positional `repo` as `owner/repo`
- Optional CLI params:
- `--format` (`table|json`)

### `scripts/list_issues.py`

- Method: `list_issues(owner: str, repo: str, state="open", per_page=100) -> list[dict]`
- Required CLI params:
- positional `repo` as `owner/repo`
- Optional CLI params:
- `--state` (`open|closed|all`)
- `--limit`
- `--format` (`table|json`)

### `scripts/list_prs.py`

- Method: `list_pull_requests(owner: str, repo: str, state="open", per_page=100) -> list[dict]`
- Required CLI params:
- positional `repo` as `owner/repo`
- Optional CLI params:
- `--state` (`open|closed|all`)
- `--limit`
- `--format` (`table|json`)

### `scripts/search_repos.py`

- Method: `search_repos(query: str, language=None, user=None, org=None, sort=None, per_page=30) -> tuple[list[dict], int]`
- Required CLI params:
- positional `query`
- Optional CLI params:
- `--language`
- `--user`
- `--org`
- `--sort` (`stars|forks|updated|best-match`)
- `--limit`
- `--format` (`table|json`)

## Usage Examples

```bash
python agent_skills/skills/github/scripts/list_repos.py --visibility private --limit 30
python agent_skills/skills/github/scripts/get_repo.py datalayer/agent-runtimes --format json
python agent_skills/skills/github/scripts/list_issues.py datalayer/agent-runtimes --state open --limit 20
python agent_skills/skills/github/scripts/list_prs.py datalayer/agent-runtimes --state all --limit 20
python agent_skills/skills/github/scripts/search_repos.py "notebook collaboration" --language python --limit 10
```

---
name: github
description: GitHub API scripts for repository discovery and inspection. Use this skill when listing your repos, inspecting one repo, or listing issues/PRs. Requires GITHUB_TOKEN.
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

Use this skill to call the GitHub scripts through `run_skill_script`.

## Required Environment Variables

- `GITHUB_TOKEN` (required)

## Invocation Contract

- Always call `run_skill_script` with:
  - `skill_name`: `github`
  - `script_name`: one of `list_repos`, `get_repo`, `list_issues`, `list_prs`, `search_repos`
- Pass positional parameters with `args` in exact order.
- Pass optional flags with `kwargs` using the documented keys below.
- Do not invent keys. Unknown keys are converted into unknown CLI flags and the script exits with code `2`.
- For flag-only options, use `args` (for this skill all options expect values, so `kwargs` is fine).

## Scripts API

### `script_name: list_repos`

- Purpose: list repositories for the authenticated user.
- Positional `args`: none.
- Supported `kwargs`:
  - `visibility`: `all | public | private`
  - `sort`: `updated | created | pushed | full_name`
  - `format`: `table | json`
  - `limit`: integer
- Not supported: `per_page` (this causes a CLI error).

### `script_name: get_repo`

- Purpose: get details of one repository.
- Positional `args`:
  - `["owner/repo"]`
- Supported `kwargs`:
  - `format`: `table | json`

### `script_name: list_issues`

- Purpose: list issues for one repository.
- Positional `args`:
  - `["owner/repo"]`
- Supported `kwargs`:
  - `state`: `open | closed | all`
  - `format`: `table | json`
  - `limit`: integer

### `script_name: list_prs`

- Purpose: list pull requests for one repository.
- Positional `args`:
  - `["owner/repo"]`
- Supported `kwargs`:
  - `state`: `open | closed | all`
  - `format`: `table | json`
  - `limit`: integer

### `script_name: search_repos`

- Purpose: search repositories by query and optional qualifiers.
- Positional `args`:
  - `["query text"]`
- Supported `kwargs`:
  - `language`: string
  - `user`: string
  - `org`: string
  - `sort`: `stars | forks | updated | best-match`
  - `format`: `table | json`
  - `limit`: integer

## `run_skill_script` Examples

- List repos (valid):

```json
{
  "skill_name": "github",
  "script_name": "list_repos",
  "kwargs": {
    "sort": "updated",
    "limit": 3,
    "format": "json"
  }
}
```

- Get one repo:

```json
{
  "skill_name": "github",
  "script_name": "get_repo",
  "args": ["datalayer/agent-runtimes"],
  "kwargs": {"format": "json"}
}
```

## Direct CLI Examples

```bash
python agent_skills/skills/github/scripts/list_repos.py --visibility private --limit 30
python agent_skills/skills/github/scripts/get_repo.py datalayer/agent-runtimes --format json
python agent_skills/skills/github/scripts/list_issues.py datalayer/agent-runtimes --state open --limit 20
python agent_skills/skills/github/scripts/list_prs.py datalayer/agent-runtimes --state all --limit 20
python agent_skills/skills/github/scripts/search_repos.py "notebook collaboration" --language python --limit 10
```

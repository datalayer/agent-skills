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
  - `skill_name`: `github` (lowercase, exactly as returned by `list_skills`)
  - `script_name`: one of `list_repos`, `get_repo`, `list_issues`, `list_prs`, `search_repos`
- Pass positional parameters with `args` in exact order.
- Pass optional flags with `kwargs` using the documented keys below.
- Do not invent keys. Unknown keys are converted into unknown CLI flags and the script exits with code `2`.
- For flag-only options, use `args` (for this skill all options expect values, so `kwargs` is fine).

## Common Failure Modes

- `Skill not found: GitHub`: use lowercase `github` as listed by `list_skills`.
- Exit code `2`: one or more `kwargs` keys do not exist in the script CLI, or a required positional arg is missing. Read the stderr output — it lists all valid parameters. Retry with only those.
- `Error: GITHUB_TOKEN environment variable is required`: the request context does not include GitHub identity/token.
- `Error: Invalid or expired GITHUB_TOKEN`: token exists but is invalid/expired.

### Recovery from exit code 2

When a script exits with code `2`, the error output includes a **"Valid parameters"** block listing every accepted flag with types and defaults. Use that list to build a corrected `kwargs`/`args` and retry. Common mistakes:
- Using `per_page` instead of `limit`.
- Passing `org` or `user` to `list_repos` (those belong to `search_repos`).
- Omitting the required positional `query` arg for `search_repos` (use `args: ["<query>"]`).

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

- List private repos only:

```json
{
  "skill_name": "github",
  "script_name": "list_repos",
  "kwargs": {
    "visibility": "private",
    "sort": "updated",
    "limit": 3
  }
}
```

- Search org repos (note: `query` is a **positional** arg via `args`, `org` is a kwarg):

```json
{
  "skill_name": "github",
  "script_name": "search_repos",
  "args": ["*"],
  "kwargs": {
    "org": "datalayer",
    "sort": "updated",
    "limit": 3
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

### Common mistakes (exit code 2)

```json
// ❌ WRONG — per_page is not a valid kwarg, use limit
{"skill_name": "github", "script_name": "list_repos", "kwargs": {"per_page": 3}}

// ✅ CORRECT
{"skill_name": "github", "script_name": "list_repos", "kwargs": {"limit": 3}}

// ❌ WRONG — search_repos requires positional query in args
{"skill_name": "github", "script_name": "search_repos", "kwargs": {"org": "datalayer"}}

// ✅ CORRECT — pass query text via args
{"skill_name": "github", "script_name": "search_repos", "args": ["*"], "kwargs": {"org": "datalayer", "limit": 3}}
```

## Direct CLI Examples

```bash
python agent_skills/skills/github/scripts/list_repos.py --visibility private --limit 30
python agent_skills/skills/github/scripts/get_repo.py datalayer/agent-runtimes --format json
python agent_skills/skills/github/scripts/list_issues.py datalayer/agent-runtimes --state open --limit 20
python agent_skills/skills/github/scripts/list_prs.py datalayer/agent-runtimes --state all --limit 20
python agent_skills/skills/github/scripts/search_repos.py "notebook collaboration" --language python --limit 10
```

---
name: events
description: Event API helper for create/get/list/update operations against the AI Agents events API. Requires DATALAYER_API_KEY.
license: Proprietary. LICENSE.txt has complete terms
version: 1.0.0
tags:
  - events
  - orchestration
  - scheduling
author: Datalayer
---

# Events Skill

## Required Environment Variables

- `DATALAYER_API_KEY` (required)
- `DATALAYER_AI_AGENTS_URL` (optional, default `https://prod1.datalayer.run`)

## Invocation Contract

- Call `run_skill_script` with:
  - `skill_name`: `events`
  - `script_name`: `event`
- This script uses subcommands, so put the subcommand in `args[0]`.
- Use `args` for required positional items (for example, `event_id`).
- Use `kwargs` for optional and named flags.

## Script API

### `script_name: event`

Subcommands and parameters:

1. `create`
- `args`: `["create"]`
- required `kwargs`: `agent_id`, `title`
- optional `kwargs`: `kind`, `status`, `payload`, `metadata`
- `payload` and `metadata` must be JSON object strings.

2. `get`
- `args`: `["get", "<event_id>"]`
- `kwargs`: none

3. `list`
- `args`: `["list"]`
- optional `kwargs`: `agent_id`, `kind`, `status`, `limit`, `offset`

4. `update`
- `args`: `["update", "<event_id>"]`
- optional `kwargs`: `title`, `kind`, `status`, `payload`, `metadata`
- `payload` and `metadata` must be JSON object strings.

## `run_skill_script` Examples

- Create:

```json
{
  "skill_name": "events",
  "script_name": "event",
  "args": ["create"],
  "kwargs": {
    "agent_id": "data-acquisition",
    "title": "New dataset",
    "kind": "dataset_ingested",
    "payload": "{\"dataset\":\"imerg\"}"
  }
}
```

- Update:

```json
{
  "skill_name": "events",
  "script_name": "event",
  "args": ["update", "evt_123"],
  "kwargs": {
    "status": "completed"
  }
}
```

## Direct CLI Examples

```bash
python agent_skills/skills/events/scripts/event.py create --agent-id data-acquisition --title "New dataset" --kind dataset_ingested --payload '{"dataset":"imerg"}'
python agent_skills/skills/events/scripts/event.py get evt_123
python agent_skills/skills/events/scripts/event.py list --agent-id data-acquisition --status pending --limit 20
python agent_skills/skills/events/scripts/event.py update evt_123 --status completed
```

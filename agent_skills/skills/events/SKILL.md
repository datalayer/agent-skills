---
name: events
description: Event API helper scripts for create/get/list/update operations. Use scripts from scripts/ and provide required event fields and auth environment variables.
license: Proprietary. LICENSE.txt has complete terms
version: 1.0.0
tags:
  - events
  - orchestration
  - scheduling
author: Datalayer
---

# Events Skill

## Environment

- `DATALAYER_API_KEY` (required)
- `DATALAYER_AI_AGENTS_URL` (optional, default `https://prod1.datalayer.run`)

## Script Inventory

### `scripts/event.py`

- Method: `create_event(agent_id, title, kind="generic", status="pending", payload=None, metadata=None)`
- Method: `get_event(event_id)`
- Method: `list_events(agent_id=None, kind=None, status=None, limit=50, offset=0)`
- Method: `update_event(event_id, title=None, kind=None, status=None, payload=None, metadata=None)`
- CLI subcommands:
- `create` required params: `--agent-id`, `--title`
- `get` required params: positional `event_id`
- `list` required params: none
- `update` required params: positional `event_id`
- Optional CLI params:
- `create`: `--kind`, `--status`, `--payload`, `--metadata`
- `list`: `--agent-id`, `--kind`, `--status`, `--limit`, `--offset`
- `update`: `--title`, `--kind`, `--status`, `--payload`, `--metadata`

## Usage Examples

```bash
python agent_skills/skills/events/scripts/event.py create --agent-id data-acquisition --title "New dataset" --kind dataset_ingested --payload '{"dataset":"imerg"}'
python agent_skills/skills/events/scripts/event.py get evt_123
python agent_skills/skills/events/scripts/event.py list --agent-id data-acquisition --status pending --limit 20
python agent_skills/skills/events/scripts/event.py update evt_123 --status completed
```

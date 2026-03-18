---
name: events
description: Event orchestration toolkit for generating, storing, and updating event records in AI agent workflows.
license: Proprietary. LICENSE.txt has complete terms
version: 1.0.0
tags:
  - events
  - orchestration
  - scheduling
author: Datalayer
---

# Events Skill

## Overview

Use this skill to create, list, inspect, and update event records that coordinate agent workflows. It is useful for event-driven automation, traceability, and handoffs between agents.

## Typical Workflow

1. Create an event when a trigger is received.
2. Add payload metadata used by downstream agents.
3. Update status as processing advances (`pending` -> `running` -> `completed` or `failed`).
4. Query events for audit and replay.

## Python Example (Direct API)

```python
import os
import httpx

BASE_URL = os.environ.get('DATALAYER_AI_AGENTS_URL', 'https://prod1.datalayer.run')
TOKEN = os.environ['DATALAYER_API_KEY']
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'Accept': 'application/json',
    'Content-Type': 'application/json',
}

# Create
create_resp = httpx.post(
    f'{BASE_URL}/api/ai-agents/v1/events',
    headers=HEADERS,
    json={
        'agent_id': 'data-acquisition',
        'title': 'New dataset available',
        'kind': 'dataset_ingested',
        'status': 'pending',
        'payload': {'dataset': 'earthdata-gpm-imerg'},
    },
    timeout=30.0,
)
create_resp.raise_for_status()
event = create_resp.json()['event']

# Update status
update_resp = httpx.patch(
    f"{BASE_URL}/api/ai-agents/v1/events/{event['id']}",
    headers=HEADERS,
    json={'status': 'completed'},
    timeout=30.0,
)
update_resp.raise_for_status()

# List events for one agent
list_resp = httpx.get(
    f'{BASE_URL}/api/ai-agents/v1/events',
    headers=HEADERS,
    params={'agent_id': 'data-acquisition', 'limit': 20},
    timeout=30.0,
)
list_resp.raise_for_status()
print(list_resp.json())
```

## Notes

- Keep payloads compact and deterministic.
- Use stable `kind` values (for example `document_uploaded`, `dataset_ingested`).
- Always update `status` so operators can track lifecycle progress.

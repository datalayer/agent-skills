# Copyright (c) 2023-2026 Datalayer, Inc.
#
# Datalayer License

"""Executable helper for the events skill.

This module provides a minimal client and a small CLI for interacting with
`/api/ai-agents/v1/events`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import httpx


API_PATH = "/api/ai-agents/v1/events"


def _base_url() -> str:
    return os.environ.get("DATALAYER_AI_AGENTS_URL", "https://prod1.datalayer.run").rstrip("/")


def _token() -> str:
    token = os.environ.get("DATALAYER_API_KEY")
    if not token:
        raise RuntimeError("DATALAYER_API_KEY environment variable is required")
    return token


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def create_event(
    agent_id: str,
    title: str,
    kind: str = "generic",
    status: str = "pending",
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response = httpx.post(
        f"{_base_url()}{API_PATH}",
        headers=_headers(),
        json={
            "agent_id": agent_id,
            "title": title,
            "kind": kind,
            "status": status,
            "payload": payload or {},
            "metadata": metadata or {},
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["event"]


def get_event(event_id: str) -> dict[str, Any]:
    response = httpx.get(
        f"{_base_url()}{API_PATH}/{event_id}",
        headers=_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["event"]


def list_events(
    *,
    agent_id: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit, "offset": offset}
    if agent_id:
        params["agent_id"] = agent_id
    if kind:
        params["kind"] = kind
    if status:
        params["status"] = status

    response = httpx.get(
        f"{_base_url()}{API_PATH}",
        headers=_headers(),
        params=params,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def update_event(
    event_id: str,
    *,
    title: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if title is not None:
        body["title"] = title
    if kind is not None:
        body["kind"] = kind
    if status is not None:
        body["status"] = status
    if payload is not None:
        body["payload"] = payload
    if metadata is not None:
        body["metadata"] = metadata

    response = httpx.patch(
        f"{_base_url()}{API_PATH}/{event_id}",
        headers=_headers(),
        json=body,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["event"]


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must decode to an object")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Events skill helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create an event")
    create_parser.add_argument("--agent-id", required=True)
    create_parser.add_argument("--title", required=True)
    create_parser.add_argument("--kind", default="generic")
    create_parser.add_argument("--status", default="pending")
    create_parser.add_argument("--payload", default="{}")
    create_parser.add_argument("--metadata", default="{}")

    get_parser = subparsers.add_parser("get", help="Get one event")
    get_parser.add_argument("event_id")

    list_parser = subparsers.add_parser("list", help="List events")
    list_parser.add_argument("--agent-id")
    list_parser.add_argument("--kind")
    list_parser.add_argument("--status")
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.add_argument("--offset", type=int, default=0)

    update_parser = subparsers.add_parser("update", help="Update one event")
    update_parser.add_argument("event_id")
    update_parser.add_argument("--title")
    update_parser.add_argument("--kind")
    update_parser.add_argument("--status")
    update_parser.add_argument("--payload")
    update_parser.add_argument("--metadata")

    args = parser.parse_args()

    if args.command == "create":
        event = create_event(
            agent_id=args.agent_id,
            title=args.title,
            kind=args.kind,
            status=args.status,
            payload=_json_arg(args.payload),
            metadata=_json_arg(args.metadata),
        )
        print(json.dumps(event, indent=2, sort_keys=True))
        return

    if args.command == "get":
        event = get_event(args.event_id)
        print(json.dumps(event, indent=2, sort_keys=True))
        return

    if args.command == "list":
        data = list_events(
            agent_id=args.agent_id,
            kind=args.kind,
            status=args.status,
            limit=args.limit,
            offset=args.offset,
        )
        print(json.dumps(data, indent=2, sort_keys=True))
        return

    if args.command == "update":
        event = update_event(
            args.event_id,
            title=args.title,
            kind=args.kind,
            status=args.status,
            payload=_json_arg(args.payload) if args.payload is not None else None,
            metadata=_json_arg(args.metadata) if args.metadata is not None else None,
        )
        print(json.dumps(event, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

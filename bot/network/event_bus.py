"""
bot/network/event_bus.py – lightweight in-process pub/sub

Events are plain dicts: { "type": "EVENT_NAME", ...payload }

Usage:
    bus = EventBus()
    bus.subscribe("MOB_SPOTTED", handler)
    bus.publish({"type": "MOB_SPOTTED", "x": 100, "y": 200})
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable, Awaitable

import config

Handler = Callable[[dict], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[str, list[Handler]] = defaultdict(list)
        # Queue bridges the network layer → scenario loop
        self._queue: asyncio.Queue[dict] = asyncio.Queue()

    # ── Subscribe / publish ───────────────────────────────────────────

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._listeners[event_type].append(handler)

    async def publish(self, event: dict) -> None:
        """Dispatch locally and push onto the queue.

        Automatically stamps ``"from"`` with the local character name
        on events that originate here (i.e. not already tagged by a
        remote sender).
        """
        if "from" not in event and event.get("_from") != "remote":
            event["from"] = config.CHARACTER_NAME
        await self._queue.put(event)
        handlers = self._listeners.get(event.get("type", ""), [])
        for h in handlers:
            asyncio.create_task(h(event))

    # ── Queue access (used by the scenario loop) ──────────────────────

    async def next_event(self) -> dict:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

"""
bot/network/client.py – WebSocket client (run on PC2)

Connects to the server, forwards local events to it, and
injects received remote events into the local bus.
"""

from __future__ import annotations

import asyncio
import json
import logging

import websockets

import config
from bot.network.event_bus import EventBus

log = logging.getLogger(__name__)

RECONNECT_DELAY = 5  # seconds between reconnect attempts


class NetworkClient:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._ws = None

    # ── Send an event to the server ───────────────────────────────────

    async def send(self, event: dict) -> None:
        if self._ws is None:
            log.warning("[Client] not connected, dropping event")
            return
        try:
            await self._ws.send(json.dumps(event))
        except websockets.ConnectionClosed:
            self._ws = None

    # ── Main loop: connect → listen, reconnect on drop ────────────────

    async def start(self) -> None:
        uri = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
        while True:
            try:
                log.info(f"[Client] connecting to {uri}")
                async with websockets.connect(uri) as ws:
                    self._ws = ws
                    log.info("[Client] connected")
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                            event["_from"] = "remote"
                            log.debug(f"[Client] received: {event}")
                            await self._bus.publish(event)
                        except json.JSONDecodeError:
                            log.warning(f"[Client] bad message: {raw!r}")
            except (OSError, websockets.WebSocketException) as exc:
                log.warning(f"[Client] connection error: {exc!r}")
            finally:
                self._ws = None
            log.info(f"[Client] reconnecting in {RECONNECT_DELAY}s …")
            await asyncio.sleep(RECONNECT_DELAY)

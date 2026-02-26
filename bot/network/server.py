"""
bot/network/server.py – WebSocket server (run on PC1)

Broadcasts every event it receives from the local bot to all
connected clients, and forwards remote events into the local bus.
"""

from __future__ import annotations

import asyncio
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

import config
from bot.network.event_bus import EventBus

log = logging.getLogger(__name__)


class NetworkServer:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._clients: set[WebSocketServerProtocol] = set()

    # ── Broadcast to all connected clients ───────────────────────────

    async def broadcast(self, event: dict) -> None:
        if not self._clients:
            return
        message = json.dumps(event)
        await asyncio.gather(
            *[ws.send(message) for ws in self._clients],
            return_exceptions=True,
        )

    # ── Per-client handler ────────────────────────────────────────────

    async def _handle(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        log.info(f"[Server] client connected: {ws.remote_address}")
        try:
            async for raw in ws:
                try:
                    event = json.loads(raw)
                    event["_from"] = "remote"
                    log.debug(f"[Server] received: {event}")
                    await self._bus.publish(event)
                except json.JSONDecodeError:
                    log.warning(f"[Server] bad message: {raw!r}")
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            log.info(f"[Server] client disconnected: {ws.remote_address}")

    # ── Start listening ────────────────────────────────────────────────

    async def start(self) -> None:
        log.info(f"[Server] listening on {config.SERVER_HOST}:{config.SERVER_PORT}")
        async with websockets.serve(self._handle, config.SERVER_HOST, config.SERVER_PORT):
            await asyncio.Future()  # run forever

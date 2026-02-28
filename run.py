"""
run.py – entry point

Usage:
    # On PC1 (acts as server):
    python run.py --role server

    # On PC2 (acts as client, connecting to PC1):
    python run.py --role client --host 192.168.1.X

    Stop the bot: press Ctrl+C in the terminal.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import mss

import config
from bot.screen import ScreenCapture, TemplateMatcher
from bot.input_handler import InputHandler
from bot.network.event_bus import EventBus
from bot.network.server import NetworkServer
from bot.network.client import NetworkClient
from bot.scenarios import ScenarioRunner, BotStopRequested

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  Main bot loop
# ─────────────────────────────────────────────────────────────────────

async def bot_loop(
    bus: EventBus,
    network_sender,           # NetworkServer | NetworkClient
) -> None:
    capture = ScreenCapture()
    matcher = TemplateMatcher()
    ih = InputHandler()
    scenarios = ScenarioRunner()

    # Focus the game window by clicking center of the game monitor
    with mss.mss() as sct:
        mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2
    log.info(f"[bot] clicking game monitor center ({cx}, {cy}) to focus")
    ih.click(cx, cy)
    await asyncio.sleep(0.3)

    # One-time camera setup
    scenarios.initialize(ih)

    active_scenarios = scenarios.get_scenarios(config.ACTIVE_SCENARIOS)

    interval = 1.0 / max(config.CAPTURE_FPS, 1)
    log.info(f"[bot] starting loop at {config.CAPTURE_FPS} FPS "
             f"with scenarios: {config.ACTIVE_SCENARIOS}")
    log.info("[bot] press Ctrl+C in the terminal to stop")

    while True:
        tick_start = time.monotonic()

        frame = capture.grab()

        # Pull all pending remote events (non-blocking)
        remote_events: list[dict] = []
        while not bus._queue.empty():
            remote_events.append(await bus.next_event())
            bus.task_done()

        # Run each active scenario once per tick
        for scenario_fn in active_scenarios:
            ev = remote_events[0] if remote_events else None
            try:
                await scenario_fn(frame, matcher, ih, bus, ev)
            except BotStopRequested as stop:
                log.info(f"[bot] stop requested: {stop}")
                return
            except Exception as exc:
                log.exception(f"[bot] scenario {scenario_fn.__name__} error: {exc}")

        # Forward locally-published events to the other PC
        while not bus._queue.empty():
            local_event = await bus.next_event()
            bus.task_done()
            if local_event.get("_from") != "remote":
                await network_sender(local_event)

        # Pace the loop
        elapsed = time.monotonic() - tick_start
        await asyncio.sleep(max(0.0, interval - elapsed))


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────

async def main(role: str, host: str) -> None:
    bus = EventBus()

    if role == "server":
        net = NetworkServer(bus)
        sender = net.broadcast
        network_task = asyncio.create_task(net.start())
    else:
        config.SERVER_HOST = host
        net = NetworkClient(bus)
        sender = net.send
        network_task = asyncio.create_task(net.start())

    bot_task = asyncio.create_task(bot_loop(bus, sender))

    try:
        await asyncio.gather(network_task, bot_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("[bot] shutting down …")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lineage 2 multi-PC bot")
    parser.add_argument(
        "--role",
        choices=["server", "client"],
        default=config.ROLE,
        help="Run as 'server' (PC1) or 'client' (PC2)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server IP address (only needed for --role client)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.SERVER_PORT,
        help="WebSocket port",
    )
    args = parser.parse_args()
    config.SERVER_PORT = args.port

    try:
        asyncio.run(main(args.role, args.host))
    except KeyboardInterrupt:
        log.info("[bot] stopped")

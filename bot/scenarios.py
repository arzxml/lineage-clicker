"""
bot/scenarios.py – game logic / scenario definitions

Each scenario is an async function that receives:
    frame         – current screen as a BGR numpy array
    matcher       – TemplateMatcher
    input_handler – InputHandler
    bus           – EventBus  (publish events other PCs should react to)
    event         – optional incoming event from remote PC (may be None)

Add your own scenarios below and register them in SCENARIO_REGISTRY.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from bot.screen import TemplateMatcher
from bot.input_handler import InputHandler
from bot.network.event_bus import EventBus

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
#  EXAMPLE SCENARIOS
# ─────────────────────────────────────────────────────────────────────


async def scenario_auto_attack(
    frame: np.ndarray,
    matcher: TemplateMatcher,
    ih: InputHandler,
    bus: EventBus,
    event: Optional[dict] = None,
) -> None:
    """
    If the 'target_hp_bar' template is not visible, press F1 to
    target nearest enemy and click to attack.
    """
    if matcher.find(frame, "target_hp_bar") is None:
        log.debug("[auto_attack] no target – searching")
        ih.press("f1")  # Lineage 2 default: target nearest
    else:
        attack_btn = matcher.find(frame, "attack_button")
        if ih.click_template(attack_btn):
            log.debug("[auto_attack] attacking")


async def scenario_loot_nearby(
    frame: np.ndarray,
    matcher: TemplateMatcher,
    ih: InputHandler,
    bus: EventBus,
    event: Optional[dict] = None,
) -> None:
    """
    If a loot bag template appears, sweep the area and notify the
    other PC that loot is available.
    """
    loot = matcher.find(frame, "loot_bag")
    if loot:
        x, y, conf = loot
        log.info(f"[loot_nearby] loot at ({x},{y}) conf={conf:.2f}")
        ih.click(x, y)
        await bus.publish({"type": "LOOT_SPOTTED", "x": x, "y": y})


async def scenario_react_to_remote_loot(
    frame: np.ndarray,
    matcher: TemplateMatcher,
    ih: InputHandler,
    bus: EventBus,
    event: Optional[dict] = None,
) -> None:
    """
    When the *other* PC spots loot and broadcasts LOOT_SPOTTED,
    this PC can move to assist or perform a different action.
    """
    if event and event.get("type") == "LOOT_SPOTTED":
        log.info(f"[react_remote_loot] partner found loot: {event}")
        # TODO: navigate to area, assist, etc.


# ─────────────────────────────────────────────────────────────────────
#  REGISTRY  –  name → scenario function
# ─────────────────────────────────────────────────────────────────────

SCENARIO_REGISTRY = {
    "auto_attack": scenario_auto_attack,
    "loot_nearby": scenario_loot_nearby,
    "react_to_remote_loot": scenario_react_to_remote_loot,
}

"""
bot/scenarios.py – game logic / scenario definitions

Scenarios are methods on the ScenarioRunner class, which holds
shared state (e.g. BotState enum) without globals.

Each scenario method receives:
    frame         – current screen as a BGR numpy array
    matcher       – TemplateMatcher
    ih            – InputHandler
    bus           – EventBus  (publish events other PCs should react to)
    event         – optional incoming event from remote PC (may be None)

Add your own scenarios below and register them in SCENARIO_REGISTRY.
"""

from __future__ import annotations

import enum
import logging
import math
import time as _time
from time import sleep
from typing import Optional

import numpy as np

from bot.screen import ScreenCapture, TemplateMatcher
from bot.input_handler import InputHandler
from bot.network.event_bus import EventBus
from bot.network.remote_state import RemoteStateStore
from bot import vision, ocr, navigation
import config

log = logging.getLogger(__name__)


class BotState(enum.Enum):
    """Granular activity states for the bot."""
    IDLE = "idle"
    MOVING = "moving"                           # walking toward a mob
    IN_RANGE = "in_range"                       # mob nearby, ready for targeting
    LOOKING_FOR_TARGET = "looking_for_target"   # post-loot, checking for nearby mobs
    LOOTING = "looting"                         # picking up drops
    LOOTING_DONE = "looting_done"               # loot finished, check for nearby mobs
    TARGET_KILLED = "target_killed"             # target just died, ready to loot
    ATTACKING = "attacking"                     # actively attacking a target
    ATTACKING_NEARBY = "attacking"              # (alias) fighting a nearby mob
    TARGET_ACQUIRED = "target_acquired"         # target just acquired, run buffs before attacking
    PRE_ORIENTING = "pre_orienting"             # camera pre-rotation
    PATROLLING = "patrolling"                   # returning to patrol zone


class BotStopRequested(Exception):
    """Raised by a scenario to signal the bot loop should exit."""


class ScenarioRunner:
    """Holds all scenarios as methods with shared instance state."""

    def __init__(self, remote_state: Optional[RemoteStateStore] = None) -> None:
        self._state = BotState.IDLE
        self._initialized = False
        self._pre_oriented = False              # set by pre_orient_to_next_mob
        self._last_patrol_check: float = 0.0    # monotonic timestamp
        self._returning_to_zone = False         # currently heading back

        # Shared store for data received from remote bots
        self.remote_state: RemoteStateStore = remote_state or RemoteStateStore()

        # Character stats (updated by read_character_stats)
        self.char_level: int = 0
        self.cp_current: int = 0
        self.cp_max: int = 0
        self.hp_current: int = 0
        self.hp_max: int = 0
        self.mp_current: int = 0
        self.mp_max: int = 0
        self.xp_percent: float = 0.0
        self._last_stats_publish: float = 0.0   # throttle network broadcasts

        # Availability is tracked for every skill in BUFF_SKILLS + COMBAT_SKILLS
        # (TOGGLE_SKILLS are always available and don't need tracking)
        tracked_names = list(config.BUFF_SKILLS) + [
            n for n in config.COMBAT_SKILLS if n not in config.BUFF_SKILLS
        ]
        self.skill_availability: dict[str, dict] = {name: {} for name in tracked_names}
        self.buff_skills_to_use: dict[str, dict] = config.BUFF_SKILLS
        self.combat_skills_to_use: dict[str, dict] = config.COMBAT_SKILLS
        self.toggle_skills_to_use: dict[str, dict] = config.TOGGLE_SKILLS

        self._last_skill_check: float = 0.0             # throttle skill window opens

        # Shared screen-capture handle (avoid re-allocating mss per call)
        self.capture = ScreenCapture()

    # ── State helpers ─────────────────────────────────────────────

    @property
    def is_idle(self) -> bool:
        return self._state == BotState.IDLE

    def _set_state(self, state: BotState) -> None:
        if self._state != state:
            log.debug(f"[state:transition] {self._state.value} → {state.value}")
        self._state = state

    def _clear_state(self) -> None:
        if self._state != BotState.IDLE:
            log.debug(f"[state:transition] {self._state.value} → idle")
        self._state = BotState.IDLE

    def _check_exit(self, matcher: TemplateMatcher) -> None:
        """Grab a fresh frame and raise BotStopRequested if exit menu is visible."""
        frame = self.capture.grab()
        if vision.is_exit_visible(frame, matcher):
            log.warning("[exit:detected] exit menu found during mid-loop check – raising stop")
            raise BotStopRequested("exit_game detected")

    # ─────────────────────────────────────────────────────────────────
    #  SCENARIOS
    # ─────────────────────────────────────────────────────────────────

    def _cancel_target(self, ih: InputHandler) -> None:
        """Cancel current target by pressing Escape."""
        ih.press("esc")

    def initialize(self, ih: InputHandler) -> None:
        """One-time setup: reset camera to top-down view."""
        if not self._initialized:
            log.info("[init:setup] one-time camera initialisation")
            self._initialized = True

    # ── Scenario methods ─────────────────────────────────────────

    async def check_mobs_in_range(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """After looting is done, check if another mob is already nearby.

        Runs in LOOTING_DONE state (set by loot_on_dead_target).
        If a mob is within close range on the minimap, transition to
        IN_RANGE so target/attack scenarios pick it up immediately
        instead of going through the full move_to_mobs walk cycle.
        """
        if self._state != BotState.LOOTING_DONE:
            return

        close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
        result = vision.find_nearest_mob_on_minimap(frame)

        if result is not None and result[2] <= close_range:
            log.debug(
                f"[check_mobs:found] mob nearby (dist={result[2]:.2f} <= {close_range}) → IN_RANGE"
            )
            self._set_state(BotState.IN_RANGE)
        else:
            dist_str = f"dist={result[2]:.2f}" if result else "none_found"
            log.debug(f"[check_mobs:none] no mob in close range ({dist_str}) → IDLE")
            self._clear_state()

    async def target_mob_in_range(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Press next-target key to acquire a target (no attack).

        Activates when IDLE or IN_RANGE and there is no current target.
        After pressing next-target, verifies acquisition.  If it fails
        and we were IN_RANGE, falls back to IDLE so move_to_mobs can
        walk closer.
        """
        if self._state not in (BotState.IDLE, BotState.IN_RANGE):
            return
        if vision.has_target(frame, matcher):
            return

        log.debug("[target:acquire] no target – pressing next-target")
        ih.press(config.KEY_NEXT_TARGET)
        sleep(0.15)

        # Verify we actually got a target
        fresh = self.capture.grab()
        if vision.has_target(fresh, matcher):
            log.debug("[target:acquire] target acquired → TARGET_ACQUIRED")
            self._set_state(BotState.TARGET_ACQUIRED)
        else:
            # next-target didn't reach anything—mob is too far
            if self._state == BotState.IN_RANGE:
                log.debug(
                    "[target:acquire] next-target failed while IN_RANGE "
                    "– falling back to IDLE so move_to_mobs can walk closer"
                )
                self._clear_state()

    async def attack_mob_in_range(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Press attack if we have a live target.

        Activates when IDLE or IN_RANGE and a target is present.
        Sets state to ATTACKING so subsequent ticks skip this scenario
        until the target dies or disappears.
        """
        # Already attacking — nothing to do until target dies
        if self._state == BotState.ATTACKING:
            return
        if self._state not in (BotState.IDLE, BotState.IN_RANGE, BotState.TARGET_ACQUIRED):
            return
        if not vision.has_target(frame, matcher):
            return
        # Don't attack a dead target (avoids double-loot cycle)
        if not vision.target_has_hp(frame):
            log.debug("[attack:skip] target HP is 0 – won't attack a dead target")
            return

        log.debug("[attack:engage] target alive with HP – pressing attack key")
        ih.press(config.KEY_ATTACK)
        self._set_state(BotState.ATTACKING)

    async def move_to_mobs(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Orient the camera and walk toward the nearest mob.

        Only activates when IDLE (after *attack_mob_in_range* failed
        to acquire a target).  After walking close enough, transitions
        to IN_RANGE so *attack_mob_in_range* picks it up next tick.
        """
        if not self.is_idle:
            return
        if vision.has_target(frame, matcher):
            return

        self._set_state(BotState.MOVING)
        try:
            log.debug("[move:start] no target visible – beginning walk toward nearest mob")
            navigation.move_to_closest_mob(
                ih,
                pre_oriented=self._pre_oriented,
                check_exit=lambda: self._check_exit(matcher),
                capture=self.capture,
            )
            self._pre_oriented = False
            # Signal attack_mob_in_range to pick up the nearby mob
            self._set_state(BotState.IN_RANGE)
        except Exception:
            self._clear_state()
            raise

    async def check_target_died(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Check if the target we are attacking has died.

        Runs while ATTACKING.  When the target's HP drops to zero,
        transitions to TARGET_KILLED so loot_on_dead_target picks it up.
        If the target disappears entirely, go back to IDLE.
        """
        if self._state != BotState.ATTACKING:
            return

        has_tgt = vision.has_target(frame, matcher)
        if not has_tgt:
            # Target gone (despawned / out of range) — back to idle
            log.debug("[combat:died] target frame disappeared (despawned/out of range) → IDLE")
            self._clear_state()
            return

        if vision.target_is_dead(frame, matcher, capture=self.capture):
            log.debug("[combat:died] target HP confirmed empty → TARGET_KILLED")
            self._set_state(BotState.TARGET_KILLED)

    async def loot_on_dead_target(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Loot the dead target.

        Activates in TARGET_KILLED state (set by check_target_died).
        After looting, transitions to LOOTING_DONE so check_mobs_in_range
        picks it up.
        """
        if self._state != BotState.TARGET_KILLED:
            return

        self._set_state(BotState.LOOTING)
        try:
            log.debug("[loot:start] target dead – beginning loot sequence")
            self._do_loot(ih, matcher)
        finally:
            self._set_state(BotState.LOOTING_DONE)

    def _do_loot(self, ih: InputHandler, matcher: Optional[TemplateMatcher] = None) -> None:
        """Press loot key several times then cancel target."""
        _loot_t0 = _time.monotonic()
        for i in range(10):
            ih.press(config.KEY_LOOT)
            sleep(0.1)
            if matcher is not None:
                self._check_exit(matcher)
        self._cancel_target(ih)
        sleep(0.1)
        log.debug(f"[loot:done] loot sequence finished ({(_time.monotonic()-_loot_t0)*1000:.0f}ms, 10 presses + cancel)")

    async def pre_orient_to_next_mob(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """While fighting, if target HP is low, pre-orient toward next mob.

        Saves several seconds between kills by doing the top-down →
        minimap-read → rotate sequence *before* the current target dies.
        """
        if self._state not in (BotState.IDLE, BotState.ATTACKING):
            return
        if self._pre_oriented:
            return
        if not vision.has_target(frame, matcher):
            return

        ratio = vision.target_hp_ratio(frame)
        threshold = getattr(config, "PRE_ORIENT_HP_THRESHOLD", 0.10)
        if ratio > threshold or ratio <= 0.01:
            # HP not low enough, or target already dead
            return

        _po_t0 = _time.monotonic()
        try:
            log.debug(
                f"[pre_orient:start] target HP low ({ratio:.3f} < {threshold:.2f}) – "
                f"rotating camera toward next mob"
            )
            # No top-down reset here — minimap works at any pitch
            # and rotate_camera_toward_mob only drags horizontally.
            frame = self.capture.grab()
            # Look for mobs beyond close range (skip the one we're fighting)
            close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
            result = vision.find_nearest_mob_on_minimap(
                frame, min_dist=close_range,
            )
            if result is None:
                log.debug("[pre_orient:abort] no next mob found on minimap (beyond close range)")
                return

            dx, dy, dist = result
            log.debug(
                f"[pre_orient:found] next mob dir=({dx:.2f},{dy:.2f}) "
                f"dist={dist:.2f} – rotating camera"
            )

            if navigation.rotate_camera_toward_mob(
                ih, mob_dist=dist,
                check_exit=lambda: self._check_exit(matcher),
                capture=self.capture,
            ):
                # Verify mob is actually roughly north before claiming success
                frame = self.capture.grab()
                check = vision.find_nearest_mob_on_minimap(
                    frame, min_dist=close_range,
                )
                if check is not None:
                    verify_angle = math.atan2(check[0], -check[1])
                    if abs(verify_angle) < 0.6:  # within ~35°
                        self._pre_oriented = True
                        log.debug(
                            f"[pre_orient:success] verified angle={math.degrees(verify_angle):.1f}° "
                            f"({(_time.monotonic()-_po_t0)*1000:.0f}ms)"
                        )
                    else:
                        log.debug(
                            f"[pre_orient:fail] verify angle={math.degrees(verify_angle):.1f}° "
                            f"exceeds 35° threshold ({(_time.monotonic()-_po_t0)*1000:.0f}ms)"
                        )
                else:
                    log.debug(f"[pre_orient:fail] mob lost during verification ({(_time.monotonic()-_po_t0)*1000:.0f}ms)")
            else:
                log.debug(f"[pre_orient:fail] mob lost during camera rotation ({(_time.monotonic()-_po_t0)*1000:.0f}ms)")
        finally:
            pass

    async def return_to_patrol_zone(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Periodically check the map; if outside patrol zone, walk back.

        The map auto-centres on the player, so the offset of the
        patrol_zone template from the map centre tells us how far
        (and in which direction) the player has drifted.
        """
        if not self.is_idle:
            return
        # Don't check while we have a live target
        if vision.has_target(frame, matcher):
            return

        interval = getattr(config, "PATROL_CHECK_INTERVAL", 30.0)
        now = _time.monotonic()
        if now - self._last_patrol_check < interval:
            return

        self._set_state(BotState.PATROLLING)
        try:
            self._last_patrol_check = now
            result = navigation.check_patrol_zone(ih, matcher, capture=self.capture)
            if result is None:
                log.debug("[patrol:check] could not locate patrol_zone on map – skipping")
                return

            dx, dy, dist = result
            threshold = getattr(config, "PATROL_MAX_DRIFT_PX", 60)
            if dist <= threshold:
                log.debug(f"[patrol:check] inside zone (drift={dist:.0f}px <= {threshold}px threshold)")
                self._returning_to_zone = False
                return

            log.info(
                f"[patrol:return] outside zone (drift={dist:.0f}px > {threshold}px) "
                f"– walking back"
            )
            self._returning_to_zone = True

            # Direction from player toward the patrol zone.
            # On the map: +dx = zone is right of player, +dy = zone is below.
            # Target angle in map-space: atan2(dx, -dy)  (0 = north, + = CW)
            target_angle = math.atan2(dx, -dy)
            log.debug(
                f"[patrol:return] facing {math.degrees(target_angle):.1f}° "
                f"toward zone – rotating camera"
            )

            navigation.rotate_camera_by_angle(ih, target_angle, capture=self.capture)

            # Walk forward toward the zone
            steps = getattr(config, "PATROL_RETURN_STEPS", 15)
            navigation.walk_in_direction(
                ih, 0.0, steps=steps,
                check_exit=lambda: self._check_exit(matcher),
                capture=self.capture,
            )

        finally:
            self._clear_state()

    # async def assist_ppl_then_attack_on_dead_or_non_existing_target(
    #     self,
    #     frame: np.ndarray,
    #     matcher: TemplateMatcher,
    #     ih: InputHandler,
    #     bus: EventBus,
    #     event: Optional[dict] = None,
    # ) -> None:
    #     if not self.is_idle:
    #         return

    #     if not vision.has_target(frame, matcher) or vision.target_is_dead(frame, matcher):
    #         self._set_state(BotState.ATTACKING_NEARBY)
    #         try:
    #             log.debug("[assist:start] no target / dead target – assisting party member")
    #             ih.press(config.KEY_TARGET_PPL)
    #             ih.press(config.KEY_ASSIST)
    #             ih.press(config.KEY_ATTACK)
    #         finally:
    #             self._clear_state()

    # async def auto_attack(
    #     self,
    #     frame: np.ndarray,
    #     matcher: TemplateMatcher,
    #     ih: InputHandler,
    #     bus: EventBus,
    #     event: Optional[dict] = None,
    # ) -> None:
    #     """Target nearest enemy and attack."""
    #     if matcher.find(frame, "target_hp_bar") is None:
    #         log.debug("[auto_attack:search] no target frame – pressing target key")
    #         ih.press("f1")
    #     else:
    #         attack_btn = matcher.find(frame, "attack_button")
    #         if ih.click_template(attack_btn):
    #             log.debug("[auto_attack:engage] clicked attack button")

    # async def loot_nearby(
    #     self,
    #     frame: np.ndarray,
    #     matcher: TemplateMatcher,
    #     ih: InputHandler,
    #     bus: EventBus,
    #     event: Optional[dict] = None,
    # ) -> None:
    #     """Click loot bags on screen."""
    #     loot = matcher.find(frame, "loot_bag")
    #     if loot:
    #         x, y, conf = loot
    #         log.info(f"[loot_nearby:found] loot bag at ({x},{y}) conf={conf:.2f} – clicking")
    #         ih.click(x, y)
    #         await bus.publish({"type": "LOOT_SPOTTED", "x": x, "y": y})

    # async def react_to_remote_loot(
    #     self,
    #     frame: np.ndarray,
    #     matcher: TemplateMatcher,
    #     ih: InputHandler,
    #     bus: EventBus,
    #     event: Optional[dict] = None,
    # ) -> None:
    #     """React when the other PC spots loot."""
    #     if event and event.get("type") == "LOOT_SPOTTED":
    #         log.info(f"[remote_loot:received] partner spotted loot: {event}")

    async def stop_if_exit_game(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Stop the bot if the in-game exit/quit menu is visible."""
        if vision.is_exit_visible(frame, matcher):
            log.warning("[exit:detected] exit menu visible – raising stop")
            raise BotStopRequested("exit_game detected")

    # ── Remote event handling ─────────────────────────────────

    async def handle_remote_events(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Unified handler for all incoming remote events.

        Routes each event to :pyattr:`remote_state` which persists
        the payload keyed by character name and event type.  Any
        scenario that needs remote data can later query
        ``self.remote_state`` instead of parsing raw events inline.
        """
        if event is None:
            return
        if event.get("_from") != "remote":
            return

        sender = event.get("from", "unknown")
        event_type = event.get("type", "UNKNOWN")
        log.debug(f"[remote:rx] {event_type} from {sender}")
        self.remote_state.update(sender, event)

    async def check_skill_availability(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Check whether tracked skills are usable on the hot bar.

        Compares each skill icon against its template (e.g. skill-Rage.png)
        in REGION_SKILL_HOT_BAR.  When a skill is on cooldown or
        unavailable the icon appears darker, so the match confidence
        drops below the threshold.

        Results are published as a SKILL_AVAILABILITY event and stored
        in ``self.skill_availability``.
        """
        if self._state not in (BotState.IDLE, BotState.ATTACKING, BotState.TARGET_ACQUIRED):
            return
        if not self.skill_availability:
            return

        interval = getattr(config, "SKILL_CHECK_INTERVAL", 10.0)
        now = _time.monotonic()
        if now - self._last_skill_check < interval:
            return
        self._last_skill_check = now

        prev = {name: info.get("available") for name, info in self.skill_availability.items()}
        current: dict[str, bool] = {}

        for skill_name in self.skill_availability:
            template_name = f"skill-{skill_name}"
            hit = matcher.find(
                frame,
                template_name,
                region=config.REGION_SKILL_HOT_BAR,
            )
            available = hit is not None
            current[skill_name] = available
            self.skill_availability[skill_name]["available"] = available
            if hit:
                log.debug(
                    f"[skill:check] {skill_name} = AVAILABLE (conf={hit[2]:.2f})"
                )
            else:
                log.debug(f"[skill:check] {skill_name} = UNAVAILABLE (below match threshold)")

        changed = current != prev
        if changed:
            log.info(f"[skill:changed] availability update: {current}")
            await bus.publish({
                "type": "SKILL_AVAILABILITY",
                "skills": current,
            })

    async def use_buff_skills(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Use buff skills after a target has been acquired.

        Runs in TARGET_ACQUIRED state.  For each skill in
        ``self.buff_skills_to_use``, checks availability and conditions,
        executes pre-actions (e.g. equip item), uses the skill, then
        executes post-actions.  Transitions to ATTACKING when done.
        """
        if self._state != BotState.TARGET_ACQUIRED:
            return

        _buff_t0 = _time.monotonic()

        for skill_name, cfg in self.buff_skills_to_use.items():
            # 1. Check availability
            avail_info = self.skill_availability.get(skill_name, {})
            if not avail_info.get("available", False):
                log.debug(f"[buff:skip] {skill_name} – not available (cooldown or missing)")
                continue

            # 2. Evaluate conditions
            conditions = cfg.get("conditions", {})
            if not self._check_buff_conditions(conditions):
                log.debug(f"[buff:skip] {skill_name} – conditions not met ({conditions})")
                continue

            # 3. Execute pre-actions
            pre = cfg.get("pre", {})
            if pre:
                log.debug(f"[buff:pre] {skill_name} – executing pre-action: {pre}")
                self._execute_buff_action(pre, frame, matcher, ih)
                sleep(0.2)

            # 4. Use the skill – find and click it on the hot bar
            template_name = f"skill-{skill_name}"
            fresh = self.capture.grab()
            hit = matcher.find(
                fresh, template_name,
                region=config.REGION_SKILL_HOT_BAR,
            )
            if hit:
                log.info(f"[buff:use] clicking {skill_name} on hotbar (conf={hit[2]:.2f})")
                ih.click(hit[0], hit[1])
                sleep(0.3)
            else:
                log.debug(f"[buff:fail] {skill_name} template not found on hotbar – skipping")
                continue

            # 5. Execute post-actions
            post = cfg.get("post", {})
            if post:
                log.debug(f"[buff:post] {skill_name} – executing post-action: {post}")
                fresh = self.capture.grab()
                self._execute_buff_action(post, fresh, matcher, ih)
                sleep(0.2)

        # Done – transition to ATTACKING (attack_mob_in_range will press attack)
        log.debug(f"[buff:done] buff phase complete ({(_time.monotonic()-_buff_t0)*1000:.0f}ms) – ready to attack")

    def _check_buff_conditions(self, conditions: dict) -> bool:
        """Evaluate buff conditions against current character stats.

        Supported conditions:
          - hp_below_percent: int  –  True if HP% < value
          - mp_below_percent: int  –  True if MP% < value
          - cp_below_percent: int  –  True if CP% < value

        Empty conditions dict → always True.
        """
        if not conditions:
            return True

        hp_below = conditions.get("hp_below_percent")
        if hp_below is not None:
            hp_pct = (self.hp_current / self.hp_max * 100) if self.hp_max > 0 else 100
            if hp_pct >= int(hp_below):
                return False

        mp_below = conditions.get("mp_below_percent")
        if mp_below is not None:
            mp_pct = (self.mp_current / self.mp_max * 100) if self.mp_max > 0 else 100
            if mp_pct >= int(mp_below):
                return False

        cp_below = conditions.get("cp_below_percent")
        if cp_below is not None:
            cp_pct = (self.cp_current / self.cp_max * 100) if self.cp_max > 0 else 100
            if cp_pct >= int(cp_below):
                return False

        return True

    def _execute_buff_action(
        self,
        action: dict,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
    ) -> None:
        """Execute a pre/post action dict.

        Supported actions:
          - equip_item: str  –  Find item-{name} on the hot bar and click it.
        """
        equip = action.get("equip_item")
        if equip:
            template_name = f"item-{equip}"
            hit = matcher.find(
                frame, template_name,
                region=config.REGION_SKILL_HOT_BAR,
            )
            if hit:
                log.info(f"[equip:click] equipping '{equip}' (conf={hit[2]:.2f})")
                ih.click(hit[0], hit[1])
                sleep(0.3)
            else:
                log.warning(f"[equip:fail] template 'item-{equip}' not found on hotbar")

    # ── OCR-based stat reading ────────────────────────────────────

    async def read_character_stats(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Read CP, HP, MP, XP and level from REGION_GENERAL_STATS.

        Delegates the heavy OCR work to ``bot.ocr.read_stats()``.
        Publishes STATS_UPDATE events over the network so other bots
        can react.
        """
        reading = ocr.read_stats(frame)
        if reading is None:
            return

        # Use previous values as fallback for fields OCR didn't detect
        level  = reading.level      if reading.level > 0      else self.char_level
        cp_cur = reading.cp_current if reading.cp_max > 0     else self.cp_current
        cp_max = reading.cp_max     if reading.cp_max > 0     else self.cp_max
        hp_cur = reading.hp_current if reading.hp_max > 0     else self.hp_current
        hp_max = reading.hp_max     if reading.hp_max > 0     else self.hp_max
        mp_cur = reading.mp_current if reading.mp_max > 0     else self.mp_current
        mp_max = reading.mp_max     if reading.mp_max > 0     else self.mp_max
        xp_pct = reading.xp_percent if reading.xp_percent > 0 else self.xp_percent

        # Check if anything changed
        changed = (
            level != self.char_level
            or cp_cur != self.cp_current or cp_max != self.cp_max
            or hp_cur != self.hp_current or hp_max != self.hp_max
            or mp_cur != self.mp_current or mp_max != self.mp_max
            or abs(xp_pct - self.xp_percent) > 0.001
        )

        self.char_level = level
        self.cp_current, self.cp_max = cp_cur, cp_max
        self.hp_current, self.hp_max = hp_cur, hp_max
        self.mp_current, self.mp_max = mp_cur, mp_max
        self.xp_percent = xp_pct

        now = _time.monotonic()
        if changed or (now - self._last_stats_publish > 5.0):
            self._last_stats_publish = now
            cp_ratio = cp_cur / cp_max if cp_max > 0 else 0.0
            hp_ratio = hp_cur / hp_max if hp_max > 0 else 0.0
            mp_ratio = mp_cur / mp_max if mp_max > 0 else 0.0
            log.debug(
                f"[stats:update] Lv{level} "
                f"CP:{cp_ratio:.0%} "
                f"HP:{hp_ratio:.0%} "
                f"MP:{mp_ratio:.0%} "
                f"XP:{xp_pct:.2f}%"
            )
            await bus.publish({
                "type": "STATS_UPDATE",
                "level": level,
                "cp_percent": round(cp_ratio * 100, 1),
                "hp_percent": round(hp_ratio * 100, 1),
                "mp_percent": round(mp_ratio * 100, 1),
                "xp_percent": round(xp_pct, 2),
            })

    # ── Camera calibration (dev tool) ─────────────────────────────

    async def calibrate_camera(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """One-shot calibration: drag known pixel amounts and log angle changes.

        Enable this scenario ALONE with a single mob visible on the
        minimap.  It will perform 5 test drags of increasing size,
        log the exact angle change for each, then stop the bot.
        Use the output to set CAMERA_PX_PER_DEG in config.
        """
        # Only run once
        if getattr(self, '_calibration_done', False):
            return
        self._calibration_done = True

        # Read Windows mouse settings for reference
        try:
            import ctypes
            speed = ctypes.c_int()
            ctypes.windll.user32.SystemParametersInfoA(112, 0, ctypes.byref(speed), 0)
            log.info(f"[calibrate] Windows mouse speed: {speed.value}/20")
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Control Panel\Mouse')
            accel, _ = winreg.QueryValueEx(key, 'MouseSpeed')
            key.Close()
            log.info(f"[calibrate] Mouse acceleration (MouseSpeed): {accel} (0=off, 1=on)")
        except Exception as e:
            log.info(f"[calibrate] Could not read mouse settings: {e}")

        cx, cy = navigation.get_screen_center(self.capture)
        test_drags = [25, 50, 100, 150, 200, -50, -100, -150, -200]

        log.info("[calibrate] === CAMERA CALIBRATION START ===")
        log.info("[calibrate] Make sure exactly ONE mob is visible on minimap")
        log.info(f"[calibrate] Screen center: ({cx}, {cy})")
        sleep(2)  # give time to read

        results = []
        for drag_px in test_drags:
            # Read angle before
            frame_before = self.capture.grab()
            r1 = vision.find_nearest_mob_on_minimap(frame_before)
            if r1 is None:
                log.warning(f"[calibrate] drag={drag_px}px: NO MOB FOUND before drag, skipping")
                sleep(1)
                continue

            angle_before = math.degrees(math.atan2(r1[0], -r1[1]))

            # Perform drag
            ih.drag_to(cx, cy, cx + drag_px, cy, duration=0.3, button="right")
            sleep(0.5)

            # Read angle after
            frame_after = self.capture.grab()
            r2 = vision.find_nearest_mob_on_minimap(frame_after)
            if r2 is None:
                log.warning(f"[calibrate] drag={drag_px}px: NO MOB FOUND after drag, skipping")
                # Undo: drag back
                ih.drag_to(cx, cy, cx - drag_px, cy, duration=0.3, button="right")
                sleep(0.5)
                continue

            angle_after = math.degrees(math.atan2(r2[0], -r2[1]))
            delta = ((angle_after - angle_before) + 180) % 360 - 180

            if abs(delta) > 0.5:
                px_per_deg = drag_px / delta
            else:
                px_per_deg = float('inf')

            results.append((drag_px, angle_before, angle_after, delta, px_per_deg))
            log.info(
                f"[calibrate] drag={drag_px:+4d}px | "
                f"before={angle_before:+7.1f}\u00b0 | "
                f"after={angle_after:+7.1f}\u00b0 | "
                f"delta={delta:+7.1f}\u00b0 | "
                f"px/deg={px_per_deg:+.2f}"
            )

            # Undo: drag back to original position
            ih.drag_to(cx, cy, cx - drag_px, cy, duration=0.3, button="right")
            sleep(0.5)

        # Summary
        valid = [r for r in results if abs(r[4]) < 1000]
        if valid:
            avg_px_per_deg = sum(r[4] for r in valid) / len(valid)
            log.info(f"[calibrate] === RESULTS ===")
            log.info(f"[calibrate] Samples: {len(valid)}")
            log.info(f"[calibrate] Average px/deg: {avg_px_per_deg:.2f}")
            log.info(f"[calibrate] Average px/rad: {avg_px_per_deg * 180 / math.pi:.2f}")
            log.info(f"[calibrate] Add to config.py:  CAMERA_PX_PER_DEG = {avg_px_per_deg:.2f}")
        else:
            log.warning("[calibrate] No valid samples collected!")

        log.info("[calibrate] === CAMERA CALIBRATION DONE ===")
        raise BotStopRequested("calibration complete")

    # ─────────────────────────────────────────────────────────────────
    #  REGISTRY  –  name → method name
    # ─────────────────────────────────────────────────────────────────

    def get_scenarios(self, names: list[str]) -> list:
        """Return a list of bound methods for the given scenario names."""
        registry = {
            # "auto_attack": self.auto_attack,
            # "loot_nearby": self.loot_nearby,
            # "assist_ppl_then_attack_on_dead_or_non_existing_target": self.assist_ppl_then_attack_on_dead_or_non_existing_target,
            "handle_remote_events": self.handle_remote_events,
            "check_target_died": self.check_target_died,
            "loot_on_dead_target": self.loot_on_dead_target,
            "pre_orient_to_next_mob": self.pre_orient_to_next_mob,
            "return_to_patrol_zone": self.return_to_patrol_zone,
            "attack_mob_in_range": self.attack_mob_in_range,
            "target_mob_in_range": self.target_mob_in_range,
            "check_mobs_in_range": self.check_mobs_in_range,
            "read_character_stats": self.read_character_stats,
            "check_skill_availability": self.check_skill_availability,
            "use_buff_skills": self.use_buff_skills,
            "move_to_mobs": self.move_to_mobs,
            "calibrate_camera": self.calibrate_camera,
            "stop_if_exit_game": self.stop_if_exit_game,
        }
        return [registry[n] for n in names if n in registry]

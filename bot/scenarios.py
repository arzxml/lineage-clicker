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

import cv2
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
        # Track which toggle skills are currently active (on)
        self._toggle_active: dict[str, bool] = {name: False for name in config.TOGGLE_SKILLS}

        self._last_skill_check: float = 0.0             # throttle skill window opens

        # Target HP stall detection – if HP ratio hasn't changed for
        # several seconds while ATTACKING, the mob is likely dead and
        # we're seeing residual red UI decoration.
        self._last_target_hp: float = -1.0
        self._target_hp_stall_since: float = 0.0

        # Periodic re-attack: re-press attack key while ATTACKING
        self._last_attack_press: float = 0.0

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
        # While ATTACKING, periodically re-press attack key in case
        # auto-attack dropped (mob moved, lag, character interrupted, etc.)
        if self._state == BotState.ATTACKING:
            interval = getattr(config, "ATTACK_REPRESS_INTERVAL", 3.0)
            now = _time.monotonic()
            if now - self._last_attack_press >= interval:
                log.debug("[attack:repress] re-pressing attack key (keep-alive)")
                ih.press(config.KEY_ATTACK)
                self._last_attack_press = now
            return
        if self._state not in (BotState.IDLE, BotState.IN_RANGE, BotState.TARGET_ACQUIRED):
            return
        if not vision.has_target(frame, matcher):
            return
        # Don't attack a dead target (avoids double-loot cycle)
        if not vision.target_has_hp(frame):
            log.debug("[attack:skip] target HP is 0 – won't attack a dead target")
            # If we just came from TARGET_ACQUIRED (buff phase) and mob died
            # in the meantime, go straight to TARGET_KILLED so loot picks it up.
            if self._state == BotState.TARGET_ACQUIRED:
                log.debug("[attack:skip] was TARGET_ACQUIRED → TARGET_KILLED (mob died during buffs)")
                self._set_state(BotState.TARGET_KILLED)
            return

        log.debug("[attack:engage] target alive with HP – pressing attack key")
        ih.press(config.KEY_ATTACK)
        self._last_attack_press = _time.monotonic()
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
        Also detects "stale HP" – if the HP ratio is unchanged for
        several seconds, the mob likely died but residual red pixels
        (UI decoration) keep the ratio non-zero.
        """
        if self._state != BotState.ATTACKING:
            return

        has_tgt = vision.has_target(frame, matcher)
        if not has_tgt:
            # Target gone (despawned / out of range) — back to idle
            log.debug("[combat:died] target frame disappeared (despawned/out of range) → IDLE")
            self._last_target_hp = -1.0
            self._clear_state()
            return

        if vision.target_is_dead(frame, matcher, capture=self.capture):
            log.debug("[combat:died] target HP confirmed empty → TARGET_KILLED")
            self._last_target_hp = -1.0
            self._set_state(BotState.TARGET_KILLED)
            return

        # ---- HP stall detection ----
        hp = vision.target_hp_ratio(frame)
        now = _time.monotonic()
        stall_timeout = getattr(config, "TARGET_HP_STALL_TIMEOUT", 4.0)

        if abs(hp - self._last_target_hp) < 0.005:
            # HP hasn't changed
            stall_dur = now - self._target_hp_stall_since
            if stall_dur >= stall_timeout:
                log.warning(
                    f"[combat:stall] target HP stuck at {hp:.3f} for "
                    f"{stall_dur:.1f}s – treating as dead → TARGET_KILLED"
                )
                self._last_target_hp = -1.0
                self._set_state(BotState.TARGET_KILLED)
                return
        else:
            # HP changed – reset stall timer
            self._last_target_hp = hp
            self._target_hp_stall_since = now

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
        """While attacking, pre-aim the camera at the next mob.

        Only activates when:
          - We are currently attacking a target.
          - We haven't already pre-oriented.
          - There are NO other mobs nearby (within close range).  If
            there are nearby mobs we'll just target one of those next,
            so camera adjustment is pointless.

        When those conditions are met, the camera is rotated toward the
        nearest *distant* mob so that when the current target dies we
        can walk straight forward instead of having to orient first.
        """
        if self._state != BotState.ATTACKING:
            return
        if self._pre_oriented:
            return
        if not vision.has_target(frame, matcher):
            return

        close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)

        # Check if there are OTHER mobs nearby (skip the one we're fighting,
        # which sits at the very centre of the minimap / very close range).
        all_mobs = vision._find_all_mobs_on_minimap(frame)
        nearby_others = [
            m for m in all_mobs
            if m[2] <= close_range and m[2] > 0.02  # > 0.02 to skip the player dot
        ]
        if nearby_others:
            # There are mobs close by — no need to rotate
            return

        # Find the nearest mob beyond close range (the next target)
        result = vision.find_nearest_mob_on_minimap(
            frame, min_dist=close_range,
        )
        if result is None:
            # No distant mobs either — nothing to aim at
            return

        # Already roughly north? Mark as pre-oriented.
        north_cone = getattr(config, "CAMERA_NORTH_THRESHOLD_DEG", 20)
        dx, dy, dist = result
        angle = math.atan2(dx, -dy)
        if abs(angle) < math.radians(north_cone):
            self._pre_oriented = True
            log.debug(
                f"[pre_orient:ok] next mob already at north "
                f"({math.degrees(angle):+.1f}°) – no rotation needed"
            )
            return

        _po_t0 = _time.monotonic()
        log.debug(
            f"[pre_orient:start] no nearby mobs – rotating toward "
            f"next mob at {math.degrees(angle):+.1f}° dist={dist:.2f}"
        )

        if navigation.rotate_camera_toward_mob(
            ih, mob_dist=dist,
            check_exit=lambda: self._check_exit(matcher),
            capture=self.capture,
        ):
            self._pre_oriented = True
            log.debug(
                f"[pre_orient:success] camera aimed at next mob "
                f"({(_time.monotonic()-_po_t0)*1000:.0f}ms)"
            )
        else:
            log.debug(
                f"[pre_orient:fail] mob lost during rotation "
                f"({(_time.monotonic()-_po_t0)*1000:.0f}ms)"
            )

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

        brightness_ratio = getattr(config, "SKILL_BRIGHTNESS_RATIO", 0.90)

        for skill_name in self.skill_availability:
            template_name = f"skill-{skill_name}"
            hit = matcher.find(
                frame,
                template_name,
                region=config.REGION_SKILL_HOT_BAR,
            )

            # Even if shape matches, check brightness to detect cooldown
            if hit is not None:
                tpl_img = matcher.get_template(template_name)
                if tpl_img is not None:
                    cx, cy, conf = hit
                    th, tw = tpl_img.shape[:2]
                    # Extract matched region from full frame
                    x1 = max(0, cx - tw // 2)
                    y1 = max(0, cy - th // 2)
                    x2 = min(frame.shape[1], x1 + tw)
                    y2 = min(frame.shape[0], y1 + th)
                    region_crop = frame[y1:y2, x1:x2]
                    # Compare mean brightness (V channel in HSV)
                    tpl_brightness = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY).mean()
                    region_brightness = cv2.cvtColor(region_crop, cv2.COLOR_BGR2GRAY).mean()
                    b_ratio = region_brightness / tpl_brightness if tpl_brightness > 0 else 1.0
                    if b_ratio < brightness_ratio:
                        log.debug(
                            f"[skill:check] {skill_name} = UNAVAILABLE "
                            f"(darkened: brightness {b_ratio:.2f} < {brightness_ratio})"
                        )
                        hit = None  # reject — skill is on cooldown

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
            use_conditions = conditions.get("use", conditions)  # support nested {"use": {...}} or flat
            if not self._check_buff_conditions(use_conditions):
                log.debug(f"[buff:skip] {skill_name} – conditions not met ({use_conditions})")
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

        # Done – press attack and transition to ATTACKING
        log.debug(f"[buff:done] buff phase complete ({(_time.monotonic()-_buff_t0)*1000:.0f}ms) – pressing attack")
        ih.press(config.KEY_ATTACK)
        self._last_attack_press = _time.monotonic()
        self._set_state(BotState.ATTACKING)
        log.debug("[buff:done] attack key sent – now ATTACKING")

    def _check_buff_conditions(self, conditions: dict) -> bool:
        """Evaluate buff conditions against current character stats.

        Supported conditions:
          - hp_below_percent: int  –  True if HP% < value
          - hp_above_percent: int  –  True if HP% > value
          - mp_below_percent: int  –  True if MP% < value
          - mp_above_percent: int  –  True if MP% > value
          - cp_below_percent: int  –  True if CP% < value
          - cp_above_percent: int  –  True if CP% > value

        Empty conditions dict → always True.
        All specified conditions must be met (AND logic).
        """
        if not conditions:
            return True

        hp_pct = (self.hp_current / self.hp_max * 100) if self.hp_max > 0 else 100
        mp_pct = (self.mp_current / self.mp_max * 100) if self.mp_max > 0 else 100
        cp_pct = (self.cp_current / self.cp_max * 100) if self.cp_max > 0 else 100

        hp_below = conditions.get("hp_below_percent")
        if hp_below is not None and hp_pct >= int(hp_below):
            return False

        hp_above = conditions.get("hp_above_percent")
        if hp_above is not None and hp_pct <= int(hp_above):
            return False

        mp_below = conditions.get("mp_below_percent")
        if mp_below is not None and mp_pct >= int(mp_below):
            return False

        mp_above = conditions.get("mp_above_percent")
        if mp_above is not None and mp_pct <= int(mp_above):
            return False

        cp_below = conditions.get("cp_below_percent")
        if cp_below is not None and cp_pct >= int(cp_below):
            return False

        cp_above = conditions.get("cp_above_percent")
        if cp_above is not None and cp_pct <= int(cp_above):
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

        # Use previous values as fallback for fields OCR didn't detect.
        # Also reject readings where current > max – that's always an
        # OCR mis-read (e.g. '/' read as '1' turning '92/440' into 921/440).
        level  = reading.level      if reading.level > 0      else self.char_level

        if reading.cp_max > 0 and reading.cp_current <= reading.cp_max:
            cp_cur, cp_max = reading.cp_current, reading.cp_max
        else:
            cp_cur, cp_max = self.cp_current, self.cp_max

        if reading.hp_max > 0 and reading.hp_current <= reading.hp_max:
            hp_cur, hp_max = reading.hp_current, reading.hp_max
        else:
            hp_cur, hp_max = self.hp_current, self.hp_max

        if reading.mp_max > 0 and reading.mp_current <= reading.mp_max:
            mp_cur, mp_max = reading.mp_current, reading.mp_max
        else:
            mp_cur, mp_max = self.mp_current, self.mp_max

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

    async def manage_toggle_skills(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Toggle skills on/off based on conditions.

        For each toggle skill:
          - If conditions are met and skill is OFF → click it to turn ON.
          - If conditions are NOT met and skill is ON → click it to turn OFF.
        """
        if not self.toggle_skills_to_use:
            return
        # Only manage toggles when we have stats (mp_max > 0)
        if self.mp_max <= 0:
            return

        for skill_name, cfg in self.toggle_skills_to_use.items():
            conditions = cfg.get("conditions", {})
            enable_conditions = conditions.get("enable", {})
            disable_conditions = conditions.get("disable", {})
            is_active = self._toggle_active.get(skill_name, False)

            log.debug(
                f"[toggle:eval] {skill_name} active={is_active} "
                f"enable_cond={enable_conditions} disable_cond={disable_conditions}"
            )

            if not is_active and enable_conditions:
                # Check if we should turn ON
                should_enable = self._check_buff_conditions(enable_conditions)
                if should_enable:
                    template_name = f"skill-{skill_name}"
                    fresh = self.capture.grab()
                    hit = matcher.find(
                        fresh, template_name,
                        region=config.REGION_SKILL_HOT_BAR,
                    )
                    if hit:
                        log.info(f"[toggle:on] enabling {skill_name} (conf={hit[2]:.2f})")
                        ih.click(hit[0], hit[1])
                        self._toggle_active[skill_name] = True
                        sleep(0.3)
                    else:
                        log.debug(f"[toggle:on] {skill_name} template not found on hotbar")

            elif is_active and disable_conditions:
                # Check if we should turn OFF
                should_disable = self._check_buff_conditions(disable_conditions)
                if should_disable:
                    template_name = f"skill-{skill_name}"
                    fresh = self.capture.grab()
                    hit = matcher.find(
                        fresh, template_name,
                        region=config.REGION_SKILL_HOT_BAR,
                    )
                    if hit:
                        log.info(f"[toggle:off] disabling {skill_name} (conf={hit[2]:.2f})")
                        ih.click(hit[0], hit[1])
                        self._toggle_active[skill_name] = False
                        sleep(0.3)
                    else:
                        log.debug(f"[toggle:off] {skill_name} template not found on hotbar")

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
            "manage_toggle_skills": self.manage_toggle_skills,
            "move_to_mobs": self.move_to_mobs,
            "calibrate_camera": self.calibrate_camera,
            "stop_if_exit_game": self.stop_if_exit_game,
        }
        return [registry[n] for n in names if n in registry]

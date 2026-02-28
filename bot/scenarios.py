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
import mss
import re as _re

from bot.screen import ScreenCapture, TemplateMatcher
from bot.input_handler import InputHandler
from bot.network.event_bus import EventBus
import config

log = logging.getLogger(__name__)

# Lazy-loaded easyocr reader (heavy import, only created once)
_ocr_reader = None

def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


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

    # Per-frame cache for _target_hp_ratio to avoid redundant computation
    # when multiple scenarios call it on the same captured frame.
    _hp_cache_frame_id: int = 0
    _hp_cache_value: float = 0.0

    def __init__(self) -> None:
        self._state = BotState.IDLE
        self._initialized = False
        self._pre_oriented = False              # set by pre_orient_to_next_mob
        self._last_patrol_check: float = 0.0    # monotonic timestamp
        self._returning_to_zone = False         # currently heading back

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

        self.skill_availability = {
            "Rage": {}, 
            "Frenzy": {},
        }

        self.buff_skills_to_use = {
            "Rage": {
                "conditions": {},
                "pre": {
                    "equip_item": "Knife",
                },
                "post": {
                    "equip_item": "Elven Long Sword",
                },
            },
            "Frenzy": {
                "conditions": {
                    "hp_below_percent": 30,
                },
                "pre": {
                    "equip_item": "Knife",
                },
                "post": {
                    "equip_item": "Elven Long Sword",
                },
            },
        }

        self._last_skill_check: float = 0.0             # throttle skill window opens

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
        """Grab a fresh frame and raise BotStopRequested if exit menu is visible.

        Call this inside long-running loops so the bot stops promptly
        even when a scenario is blocking the main tick loop.
        """
        capture = ScreenCapture()
        frame = capture.grab()
        if matcher.find(frame, "exit_game", region=config.REGION_GENERAL_MENU) is not None:
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

    # ── Target HP helpers (colour-based, transparency-proof) ──────

    @staticmethod
    def _has_target(frame: np.ndarray, matcher: TemplateMatcher) -> bool:
        """Return True if the target frame is visible (template-based)."""
        return matcher.find(frame, "has_target", region=config.REGION_TARGET) is not None

    @staticmethod
    def _target_hp_ratio(frame: np.ndarray) -> float:
        """Return the red fill ratio (0.0–1.0) of the target HP bar.

        Works by counting reddish pixels in the thin HP-bar strip.
        Immune to background transparency changes because we look for
        the *hue* of the fill, not exact pixel values.

        Results are cached per frame object so multiple scenarios
        calling this on the same tick don't recompute / double-log.
        """
        fid = id(frame)
        if ScenarioRunner._hp_cache_frame_id == fid:
            return ScenarioRunner._hp_cache_value

        rx, ry, rw, rh = config.REGION_TARGET_HP_BAR
        roi = frame[ry:ry + rh, rx:rx + rw]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red wraps around H=0/180 in OpenCV's 0-180 hue range
        # Only match BRIGHT red fill — high saturation + value
        # (excludes the dark brownish bar background)
        # Range 1: deep red  (0-10)
        lo1 = np.array([0, 120, 100], dtype=np.uint8)
        hi1 = np.array([10, 255, 255], dtype=np.uint8)
        # Range 2: red-magenta (165-180)
        lo2 = np.array([165, 120, 100], dtype=np.uint8)
        hi2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

        red_pixels = np.count_nonzero(mask)
        total = mask.size
        ratio = red_pixels / total if total else 0.0
        log.debug(f"[target:hp] red_fill={ratio:.3f} ({red_pixels}/{total} px)")

        ScenarioRunner._hp_cache_frame_id = fid
        ScenarioRunner._hp_cache_value = ratio
        return ratio

    @staticmethod
    def _target_has_hp(frame: np.ndarray) -> bool:
        """Return True if the target's HP bar has any red fill (>1%)."""
        return ScenarioRunner._target_hp_ratio(frame) > 0.01

    @staticmethod
    def _target_is_dead(frame: np.ndarray, matcher: TemplateMatcher) -> bool:
        """Target is visible but its HP bar is empty -> dead.

        Uses a confirmation re-check to avoid declaring a mob dead
        when just a sliver of HP remains.
        """
        if not ScenarioRunner._has_target(frame, matcher):
            return False
        if ScenarioRunner._target_has_hp(frame):
            return False
        # First check says dead — wait briefly and confirm
        _t0 = _time.monotonic()
        sleep(0.15)
        capture = ScreenCapture()
        frame2 = capture.grab()
        if not ScenarioRunner._has_target(frame2, matcher):
            log.debug(f"[target:dead_check] target vanished during confirm ({(_time.monotonic()-_t0)*1000:.0f}ms)")
            return False  # target disappeared entirely, not dead
        is_dead = not ScenarioRunner._target_has_hp(frame2)
        log.debug(f"[target:dead_check] confirmed={'DEAD' if is_dead else 'ALIVE'} ({(_time.monotonic()-_t0)*1000:.0f}ms)")
        return is_dead

    # def _set_top_down_view(self, ih: InputHandler) -> None:
    #     """Reset camera pitch to top-down."""
    #     with mss.mss() as sct:
    #         mon = sct.monitors[config.MONITOR_INDEX]
    #     cx = mon["left"] + mon["width"] // 2
    #     cy = mon["top"] + mon["height"] // 2
    #     # Drag DOWN hard to guarantee hitting the top-down limit
    #     ih.drag_to(cx, cy, cx, cy + 600, duration=0.8, button="right")
    #     sleep(0.2)

    # def _set_angled_view(self, ih: InputHandler) -> None:
    #     """From top-down, tilt slightly to get a playable angled view."""
    #     with mss.mss() as sct:
    #         mon = sct.monitors[config.MONITOR_INDEX]
    #     cx = mon["left"] + mon["width"] // 2
    #     cy = mon["top"] + mon["height"] // 2
    #     # Drag UP slightly — from ground/top-down toward angled
    #     tilt = getattr(config, "CAMERA_TILT_UP_PX", 40)
    #     ih.drag_to(cx, cy, cx, cy - tilt, duration=0.4, button="right")
    #     sleep(0.15)

    # ── Patrol-zone helpers ────────────────────────────────────────

    def _check_patrol_zone(
        self, ih: InputHandler, matcher: TemplateMatcher,
    ) -> Optional[tuple[float, float, float]]:
        """Open the map, find the patrol_zone template, return offset.

        Returns (dx, dy, dist) where dx/dy are pixel offsets of the
        patrol zone template centre relative to the map centre
        (positive dx = patrol zone is to the right = player is left of it),
        and *dist* is the Euclidean pixel distance.  None if template
        not found.
        """
        capture = ScreenCapture()

        # Open map
        ih.hotkey("alt", "m")
        sleep(0.6)

        frame = capture.grab()
        rx, ry, rw, rh = config.REGION_MAP
        hit = matcher.find(frame, "patrol_zone", region=config.REGION_MAP)

        # Close map immediately
        ih.hotkey("alt", "m")
        sleep(0.3)

        if hit is None:
            log.debug("[patrol:map] patrol_zone template not found on map")
            return None

        px, py, conf = hit
        # Map centre in screen coords
        map_cx = rx + rw // 2
        map_cy = ry + rh // 2
        dx = px - map_cx
        dy = py - map_cy
        dist = math.hypot(dx, dy)
        log.debug(
            f"[patrol:map] zone_offset dx={dx:.0f} dy={dy:.0f} "
            f"dist={dist:.0f}px conf={conf:.2f}"
        )
        return dx, dy, dist

    def _walk_in_direction(
        self, ih: InputHandler, angle_rad: float, steps: int = 15,
        matcher: Optional[TemplateMatcher] = None,
    ) -> None:
        """Click-walk in *angle_rad* direction (0 = north, + = CW).

        Uses screen-centre + offset to produce directional clicks.
        """
        walk_radius = getattr(config, "MOVE_FORWARD_CLICK_PX", 250)
        capture = ScreenCapture()

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        scr_cx = mon["left"] + mon["width"] // 2
        scr_cy = mon["top"] + mon["height"] // 2

        # angle_rad: 0=north (+screen-up), positive=clockwise
        click_x = scr_cx + int(walk_radius * math.sin(angle_rad))
        click_y = scr_cy - int(walk_radius * math.cos(angle_rad))

        close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
        for i in range(steps):
            ih.click(click_x, click_y)
            sleep(0.45)
            # If a mob appears nearby while returning, stop early
            frame = capture.grab()
            result = self._find_nearest_mob_on_minimap(frame)
            if result is not None and result[2] <= close_range:
                log.debug(f"[patrol:walk] mob appeared nearby (dist={result[2]:.2f}), aborting walk at step {i+1}/{steps}")
                break
            if matcher is not None:
                self._check_exit(matcher)
            if i % 5 == 4:
                log.debug(f"[patrol:walk] step {i+1}/{steps} – no mobs nearby yet")

    def _rotate_camera_toward_mob(
        self, ih: InputHandler, mob_dist: Optional[float] = None,
        matcher: Optional[TemplateMatcher] = None,
    ) -> bool:
        """Rotate camera so the target mob is directly north on the minimap.

        Uses the static CAMERA_PX_PER_RAD constant (calibrated once) to
        compute exact drag amounts.  Up to 4 correction passes to
        converge, no runtime calibration needed.

        *mob_dist*: normalised distance of the target mob from the
        minimap centre, used to re-identify the same mob after the
        camera moves (avoids confusing multiple mobs).
        """
        capture = ScreenCapture()
        px_per_rad = getattr(config, "CAMERA_PX_PER_RAD", 19.0)

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2

        _rot_t0 = _time.monotonic()
        for i in range(6):
            frame = capture.grab()
            r = self._find_nearest_mob_on_minimap(frame, target_dist=mob_dist)
            if r is None:
                log.debug(f"[camera:rotate] pass {i+1}/6 – mob lost on minimap ({(_time.monotonic()-_rot_t0)*1000:.0f}ms elapsed)")
                return False

            dx, dy, dist = r
            angle = math.atan2(dx, -dy)   # 0 = north, + = clockwise
            log.debug(
                f"[camera:rotate] pass {i+1}/6 – mob_angle={math.degrees(angle):+.1f}° "
                f"dx={dx:.3f} dy={dy:.3f} dist={dist:.2f}"
            )

            if abs(angle) < 0.17:         # ~10° — acceptable deviation
                log.debug(f"[camera:rotate] converged in {i+1} passes ({(_time.monotonic()-_rot_t0)*1000:.0f}ms)")
                return True

            # Small incremental correction: 40% of computed drag per pass
            correction = int(angle * px_per_rad * 0.4)
            correction = max(-150, min(150, correction))
            if correction == 0:
                correction = 1 if angle > 0 else -1
            log.debug(f"[camera:rotate] dragging {correction:+d}px (40% of {int(angle * px_per_rad)}px)")
            ih.drag_to(cx, cy, cx + correction, cy, duration=0.15, button="right")
            sleep(0.2)
            if matcher is not None:
                self._check_exit(matcher)

        log.debug(f"[camera:rotate] max 6 passes reached, best effort ({(_time.monotonic()-_rot_t0)*1000:.0f}ms)")
        return True

    # ── Minimap helpers ───────────────────────────────────────────

    def _find_nearest_mob_on_minimap(
        self,
        frame: np.ndarray,
        target_dist: Optional[float] = None,
        min_dist: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        """
        Detect red dots (mobs) on the minimap and return
        (dx, dy, dist) where dx/dy is the direction vector
        normalised to [-1, 1] and dist is the normalised distance
        (0 = center, 1 = edge).  Returns None if no mobs found.

        If *target_dist* is given, prefer the mob whose normalised
        distance from centre is closest to that value (used to track
        the same mob across camera rotations).
        """
        rx, ry, rw, rh = config.REGION_MINIMAP
        minimap = frame[ry:ry + rh, rx:rx + rw]

        # Circular mask – the minimap is round
        h, w = minimap.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        # Detect red pixels in HSV  (red wraps around H=0/180)
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        # Lower red range
        lo1 = np.array([0, 100, 80], dtype=np.uint8)
        hi1 = np.array([10, 255, 255], dtype=np.uint8)
        # Upper red range
        lo2 = np.array([165, 100, 80], dtype=np.uint8)
        hi2 = np.array([180, 255, 255], dtype=np.uint8)
        red_mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
        red_mask = cv2.bitwise_and(red_mask, mask)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        # Find contours of red blobs
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        best_score = float("inf")
        best_dx, best_dy, best_norm_dist = 0.0, 0.0, 0.0
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            mx = int(M["m10"] / M["m00"])
            my = int(M["m01"] / M["m00"])
            cdx = mx - center_x
            cdy = my - center_y
            dist_px = math.hypot(cdx, cdy)
            # Skip the very center (that's the player, not a mob)
            if dist_px < 5:
                continue
            norm_dx = cdx / radius
            norm_dy = cdy / radius
            norm_dist = math.hypot(norm_dx, norm_dy)

            # Skip mobs closer than min_dist (e.g. the one we're fighting)
            if min_dist is not None and norm_dist < min_dist:
                continue

            if target_dist is not None:
                # Pick mob whose distance from centre best matches target
                score = abs(norm_dist - target_dist)
            else:
                # Pick closest to centre
                score = dist_px

            if score < best_score:
                best_score = score
                best_dx, best_dy = norm_dx, norm_dy
                best_norm_dist = norm_dist

        if best_score == float("inf"):
            return None
        return best_dx, best_dy, best_norm_dist


    def _move_to_closest_mob(self, ih: InputHandler, matcher: Optional[TemplateMatcher] = None) -> None:
        """
        Phase 1 – ORIENT (top-down, one-time)
            • Reset camera to top-down
            • Iteratively rotate until mob is directly north on minimap
        Phase 2 – WALK (angled view, loop)
            • Tilt to playable angle
            • Click forward (fixed top-mid) repeatedly
            • Only check minimap *distance* — do NOT re-orient
        """
        close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
        forward_px = getattr(config, "MOVE_FORWARD_CLICK_PX", 250)
        capture = ScreenCapture()

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        scr_cx = mon["left"] + mon["width"] // 2
        scr_cy = mon["top"] + mon["height"] // 2

        # Fixed click position: top-mid of screen (always "forward")
        walk_x = scr_cx
        walk_y = scr_cy - forward_px

        _move_t0 = _time.monotonic()

        # ── Phase 1: orient ─────────────────────────────────────
        if self._pre_oriented:
            log.debug("[move:orient] skipping – already pre-oriented from pre_orient_to_next_mob")
            self._pre_oriented = False
        else:
            # Normal case: find mob on minimap and rotate toward it
            frame = capture.grab()
            result = self._find_nearest_mob_on_minimap(frame)
            if result is None:
                log.debug("[move:orient] no mobs detected on minimap – aborting")
                return

            dx, dy, dist = result
            if dist <= close_range:
                log.debug(f"[move:orient] mob already in range (dist={dist:.2f} <= {close_range})")
                return

            log.debug(f"[move:orient] nearest mob at dir=({dx:.2f},{dy:.2f}) dist={dist:.2f} – rotating camera")

            # Rotate camera until mob is roughly north
            if not self._rotate_camera_toward_mob(ih, mob_dist=dist, matcher=matcher):
                log.debug(f"[move:orient] mob lost during camera rotation – aborting ({(_time.monotonic()-_move_t0)*1000:.0f}ms)")
                return

        log.debug(f"[move:orient] orient phase done ({(_time.monotonic()-_move_t0)*1000:.0f}ms) – starting walk")

        _walk_t0 = _time.monotonic()
        max_clicks = 40
        for i in range(max_clicks):
            ih.click(walk_x, walk_y)
            sleep(0.45)

            # Check distance only — no camera changes
            frame = capture.grab()
            result = self._find_nearest_mob_on_minimap(frame)
            if result is None:
                log.debug(f"[move:walk] mob lost on minimap at step {i+1}/{max_clicks} ({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)")
                break

            _, _, dist = result
            if dist <= close_range:
                log.debug(f"[move:walk] mob in range (dist={dist:.2f}) after {i+1} steps ({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)")
                break

            if i % 5 == 4:
                log.debug(f"[move:walk] step {i+1}/{max_clicks} – dist={dist:.2f} ({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)")
            if matcher is not None:
                self._check_exit(matcher)

        log.debug(f"[move:total] move_to_closest_mob finished ({(_time.monotonic()-_move_t0)*1000:.0f}ms total)")

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
        result = self._find_nearest_mob_on_minimap(frame)

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
        if self._has_target(frame, matcher):
            return

        log.debug("[target_mob_in_range] no target – pressing next-target")
        ih.press(config.KEY_NEXT_TARGET)
        sleep(0.15)

        # Verify we actually got a target
        capture = ScreenCapture()
        fresh = capture.grab()
        if self._has_target(fresh, matcher):
            log.debug("[target_mob_in_range] target acquired – transitioning to TARGET_ACQUIRED")
            self._set_state(BotState.TARGET_ACQUIRED)
        else:
            # next-target didn’t reach anything—mob is too far
            if self._state == BotState.IN_RANGE:
                log.debug(
                    "[target_mob_in_range] next-target failed while IN_RANGE "
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
        if not self._has_target(frame, matcher):
            return
        # Don’t attack a dead target (avoids double-loot cycle)
        if not self._target_has_hp(frame):
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
        if self._has_target(frame, matcher):
            return

        self._set_state(BotState.MOVING)
        try:
            log.debug("[move:start] no target visible – beginning walk toward nearest mob")
            self._move_to_closest_mob(ih, matcher)
            # Signal attack_mob_in_range to pick up the nearby mob
            self._set_state(BotState.IN_RANGE)
        except Exception:
            self._clear_state()
            raise

    async def assist_ppl_then_attack_on_dead_or_non_existing_target(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        if not self.is_idle:
            return

        if not self._has_target(frame, matcher) or self._target_is_dead(frame, matcher):
            self._set_state(BotState.ATTACKING_NEARBY)
            try:
                log.debug("[assist:start] no target / dead target – assisting party member")
                ih.press(config.KEY_TARGET_PPL)
                ih.press(config.KEY_ASSIST)
                ih.press(config.KEY_ATTACK)
            finally:
                self._clear_state()

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

        has_target = self._has_target(frame, matcher)
        if not has_target:
            # Target gone (despawned / out of range) — back to idle
            log.debug("[combat:died] target frame disappeared (despawned/out of range) → IDLE")
            self._clear_state()
            return

        if self._target_is_dead(frame, matcher):
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
        if not self._has_target(frame, matcher):
            return

        ratio = self._target_hp_ratio(frame)
        threshold = getattr(config, "PRE_ORIENT_HP_THRESHOLD", 0.10)
        if ratio > threshold or ratio <= 0.01:
            # HP not low enough, or target already dead
            return

        # self._set_state(BotState.PRE_ORIENTING)
        _po_t0 = _time.monotonic()
        try:
            log.debug(
                f"[pre_orient:start] target HP low ({ratio:.3f} < {threshold:.2f}) – "
                f"rotating camera toward next mob"
            )
            # No top-down reset here — minimap works at any pitch
            # and _rotate_camera_toward_mob only drags horizontally.
            capture = ScreenCapture()
            frame = capture.grab()
            # Look for mobs beyond close range (skip the one we're fighting)
            close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
            result = self._find_nearest_mob_on_minimap(
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

            if self._rotate_camera_toward_mob(ih, mob_dist=dist, matcher=matcher):
                # Verify mob is actually roughly north before claiming success
                frame = capture.grab()
                close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
                check = self._find_nearest_mob_on_minimap(
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
            # self._clear_state()

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
        if self._has_target(frame, matcher):
            return

        interval = getattr(config, "PATROL_CHECK_INTERVAL", 30.0)
        now = _time.monotonic()
        if now - self._last_patrol_check < interval:
            return

        self._set_state(BotState.PATROLLING)
        try:
            self._last_patrol_check = now
            result = self._check_patrol_zone(ih, matcher)
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
            # Map axes: right = east, down = south.
            # Camera north on minimap corresponds to screen-up.
            # We need to rotate camera so that walking "forward" (north on
            # minimap / up on screen) takes us toward the zone.
            # Target angle in map-space: atan2(dx, -dy)  (0 = north, + = CW)
            target_angle = math.atan2(dx, -dy)
            log.debug(
                f"[patrol:return] facing {math.degrees(target_angle):.1f}° "
                f"toward zone – rotating camera"
            )

            # Use self-calibrating rotation via a virtual "mob" at that angle
            # on the minimap.  We place a synthetic reading and rotate toward it.
            # Simplest approach: just rotate camera by target_angle using
            # the same calibration technique.
            self._rotate_camera_by_angle(ih, target_angle)

            # Walk forward toward the zone
            steps = getattr(config, "PATROL_RETURN_STEPS", 15)
            self._walk_in_direction(ih, 0.0, steps=steps, matcher=matcher)  # 0 = straight ahead

        finally:
            self._clear_state()

    def _rotate_camera_by_angle(
        self, ih: InputHandler, target_angle: float,
    ) -> None:
        """Rotate camera by *target_angle* radians using the static calibration constant."""
        if abs(target_angle) < 0.12:  # < ~7°, not worth rotating
            return

        px_per_rad = getattr(config, "CAMERA_PX_PER_RAD", 19.0)

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2

        correction = int(target_angle * px_per_rad)
        correction = max(-300, min(300, correction))
        log.debug(
            f"[patrol:rotate] angle={math.degrees(target_angle):.1f}° "
            f"correction={correction}px"
        )
        ih.drag_to(cx, cy, cx + correction, cy, duration=0.25, button="right")
        sleep(0.3)

    async def auto_attack(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Target nearest enemy and attack."""
        if matcher.find(frame, "target_hp_bar") is None:
            log.debug("[auto_attack:search] no target frame – pressing target key")
            ih.press("f1")
        else:
            attack_btn = matcher.find(frame, "attack_button")
            if ih.click_template(attack_btn):
                log.debug("[auto_attack:engage] clicked attack button")

    async def loot_nearby(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Click loot bags on screen."""
        loot = matcher.find(frame, "loot_bag")
        if loot:
            x, y, conf = loot
            log.info(f"[loot_nearby:found] loot bag at ({x},{y}) conf={conf:.2f} – clicking")
            ih.click(x, y)
            await bus.publish({"type": "LOOT_SPOTTED", "x": x, "y": y})

    async def react_to_remote_loot(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """React when the other PC spots loot."""
        if event and event.get("type") == "LOOT_SPOTTED":
            log.info(f"[remote_loot:received] partner spotted loot: {event}")

    async def stop_if_exit_game(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Stop the bot if the in-game exit/quit menu is visible."""
        if matcher.find(frame, "exit_game", region=config.REGION_GENERAL_MENU) is not None:
            log.warning("[exit:detected] exit menu visible – raising stop")
            raise BotStopRequested("exit_game detected")

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

        capture = ScreenCapture()
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
            fresh = capture.grab()
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
                fresh = capture.grab()
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

    async def read_character_stats(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        """Read CP, HP, MP, XP and level from REGION_GENERAL_STATS.

        Uses OCR on the full stats block.  Publishes STATS_UPDATE
        events over the network so other bots can react.
        """
        rx, ry, rw, rh = config.REGION_GENERAL_STATS
        roi = frame[ry:ry + rh, rx:rx + rw]

        # Upscale for better OCR accuracy
        scale = 3
        roi_big = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)

        # Isolate white/bright text (numbers are white on coloured bars)
        b, g, r_ch = cv2.split(roi_big)
        white_mask = cv2.bitwise_and(
            cv2.bitwise_and(
                cv2.threshold(r_ch, 160, 255, cv2.THRESH_BINARY)[1],
                cv2.threshold(g, 160, 255, cv2.THRESH_BINARY)[1],
            ),
            cv2.threshold(b, 160, 255, cv2.THRESH_BINARY)[1],
        )
        # Pad for OCR edge handling
        padded = cv2.copyMakeBorder(white_mask, 20, 20, 20, 20,
                                     cv2.BORDER_CONSTANT, value=0)

        try:
            reader = _get_ocr_reader()
            results = reader.readtext(
                padded,
                allowlist='0123456789/.%CPHMXcphmpx ',
                detail=0,
                paragraph=False,
            )
            raw_lines = [s.strip() for s in results if s.strip()]
        except Exception as e:
            log.debug(f"[stats:error] OCR failed: {e}")
            return

        raw_text = ' '.join(raw_lines)
        log.debug(f"[stats:ocr] raw tokens: {raw_lines}")

        # ── Helper: extract current/max from a text chunk ──
        def _parse_pair(text: str) -> Optional[tuple[int, int]]:
            """Parse 'current/max' or 'currentmax' from text."""
            # Strip non-digit/slash chars
            clean = _re.sub(r'[^0-9/]', '', text)
            # Try with slash
            m = _re.search(r'(\d+)/(\d+)', clean)
            if m:
                return int(m.group(1)), int(m.group(2))
            # Fallback: no slash — digits may be concatenated
            digits = _re.sub(r'\D', '', clean)
            if len(digits) >= 2:
                if len(digits) % 2 == 0:
                    half = len(digits) // 2
                    return int(digits[:half]), int(digits[half:])
                # Odd length: '/' likely misread as '1' in the middle
                # Try splitting around the centre digit
                mid = len(digits) // 2
                left, right = digits[:mid], digits[mid + 1:]
                if left and right and left == right:
                    return int(left), int(right)
                # Also try mid+1 split (noise digit could be before centre)
                left2, right2 = digits[:mid + 1], digits[mid + 1:]
                if left2 and right2:
                    return int(left2), int(right2)
            return None

        # ── Label-aware parsing ──
        # OCR gives items like: ['xhmm5', 'CP', '1274 /1274', 'HP', '1235/2143', 'MP', '4401440', '8', '97%']
        # Find label positions, then grab the next item as that stat's value
        level = self.char_level
        cp_cur, cp_max = self.cp_current, self.cp_max
        hp_cur, hp_max = self.hp_current, self.hp_max
        mp_cur, mp_max = self.mp_current, self.mp_max
        xp_pct = self.xp_percent

        # Normalise labels for matching (OCR sometimes returns lowercase)
        upper_lines = [s.upper().strip() for s in raw_lines]

        def _find_label_value(label: str) -> Optional[str]:
            """Find a label in OCR results and return the next item."""
            for i, item in enumerate(upper_lines):
                if item == label and i + 1 < len(upper_lines):
                    return raw_lines[i + 1]
                # Label might be embedded: "CP 1274/1274"
                if item.startswith(label) and len(item) > len(label):
                    return item[len(label):]
            return None

        cp_text = _find_label_value('CP')
        hp_text = _find_label_value('HP')
        mp_text = _find_label_value('MP')

        if cp_text:
            pair = _parse_pair(cp_text)
            if pair:
                cp_cur, cp_max = pair
        if hp_text:
            pair = _parse_pair(hp_text)
            if pair:
                hp_cur, hp_max = pair
        if mp_text:
            pair = _parse_pair(mp_text)
            if pair:
                mp_cur, mp_max = pair

        # Level: first item often contains it (e.g. 'xhmm5' → garbage,
        # but sometimes '38').  Look for a small standalone number
        # before the first label.
        first_label_idx = len(raw_lines)
        for i, item in enumerate(upper_lines):
            if item in ('CP', 'HP', 'MP'):
                first_label_idx = i
                break
        for i in range(first_label_idx):
            lm = _re.search(r'(\d{1,3})', raw_lines[i])
            if lm:
                val = int(lm.group(1))
                if 1 <= val <= 99:  # reasonable level range
                    level = val
                    break

        # XP: look for percentage pattern in the tail items
        # Could be split across items like ['8', '97%'] → "8.97%"
        tail = ' '.join(raw_lines[first_label_idx:])
        # Try "N.NN%" first
        xp_m = _re.search(r'(\d+\.\d+)\s*%', tail)
        if xp_m:
            xp_pct = float(xp_m.group(1))
        else:
            # Try split digits + percentage: "8" + "97%" → 8.97
            xp_parts = _re.findall(r'(\d+)\s*%', tail)
            xp_digits = _re.findall(r'\b(\d{1,2})\b', tail)
            if xp_parts and len(xp_digits) >= 2:
                # Last part with % is the decimal, one before is integer
                try:
                    xp_pct = float(f"{xp_digits[-2]}.{xp_parts[-1]}")
                except (ValueError, IndexError):
                    pass

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

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2

        capture = ScreenCapture()
        test_drags = [25, 50, 100, 150, 200, -50, -100, -150, -200]

        log.info("[calibrate] === CAMERA CALIBRATION START ===")
        log.info("[calibrate] Make sure exactly ONE mob is visible on minimap")
        log.info(f"[calibrate] Screen center: ({cx}, {cy})")
        sleep(2)  # give time to read

        results = []
        for drag_px in test_drags:
            # Read angle before
            frame_before = capture.grab()
            r1 = self._find_nearest_mob_on_minimap(frame_before)
            if r1 is None:
                log.warning(f"[calibrate] drag={drag_px}px: NO MOB FOUND before drag, skipping")
                sleep(1)
                continue

            angle_before = math.degrees(math.atan2(r1[0], -r1[1]))

            # Perform drag
            ih.drag_to(cx, cy, cx + drag_px, cy, duration=0.3, button="right")
            sleep(0.5)

            # Read angle after
            frame_after = capture.grab()
            r2 = self._find_nearest_mob_on_minimap(frame_after)
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
            "auto_attack": self.auto_attack,
            "loot_nearby": self.loot_nearby,
            "react_to_remote_loot": self.react_to_remote_loot,
            "assist_ppl_then_attack_on_dead_or_non_existing_target": self.assist_ppl_then_attack_on_dead_or_non_existing_target,
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

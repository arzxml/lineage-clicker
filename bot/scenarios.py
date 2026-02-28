"""
bot/scenarios.py – game logic / scenario definitions

Scenarios are methods on the ScenarioRunner class, which holds
shared state (e.g. busy flag) without globals.

Each scenario method receives:
    frame         – current screen as a BGR numpy array
    matcher       – TemplateMatcher
    ih            – InputHandler
    bus           – EventBus  (publish events other PCs should react to)
    event         – optional incoming event from remote PC (may be None)

Add your own scenarios below and register them in SCENARIO_REGISTRY.
"""

from __future__ import annotations

import logging
import math
import time as _time
from time import sleep
from typing import Optional

import cv2
import numpy as np
import mss

from bot.screen import ScreenCapture, TemplateMatcher
from bot.input_handler import InputHandler
from bot.network.event_bus import EventBus
import config

log = logging.getLogger(__name__)


class BotStopRequested(Exception):
    """Raised by a scenario to signal the bot loop should exit."""


class ScenarioRunner:
    """Holds all scenarios as methods with shared instance state."""

    def __init__(self) -> None:
        self.busy = False
        self._initialized = False
        self._pre_oriented = False              # set by pre_orient_to_next_mob
        self._last_patrol_check: float = 0.0    # monotonic timestamp
        self._returning_to_zone = False         # currently heading back

    # ─────────────────────────────────────────────────────────────────
    #  SCENARIOS
    # ─────────────────────────────────────────────────────────────────
    def _cancel_target(self, ih: InputHandler) -> None:
        """Cancel current target by pressing Escape."""
        ih.press("esc")

    def initialize(self, ih: InputHandler) -> None:
        """One-time setup: reset camera to top-down view."""
        if not self._initialized:
            log.info("[init] setting initial top-down camera view")
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
        """
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
        log.debug(f"[hp_bar] red fill ratio: {ratio:.3f} ({red_pixels}/{total})")

        # ── DEBUG: save roi + mask once per session for tuning ──
        if not hasattr(ScenarioRunner, "_hp_debug_saved"):
            ScenarioRunner._hp_debug_saved = True
            cv2.imwrite("debug_hp_roi.png", roi)
            cv2.imwrite("debug_hp_mask.png", mask)
            h, s, v = cv2.split(hsv)
            log.debug(
                f"[hp_bar DEBUG] H min={h.min()} max={h.max()} "
                f"S min={s.min()} max={s.max()} "
                f"V min={v.min()} max={v.max()}"
            )

        return ratio

    @staticmethod
    def _target_has_hp(frame: np.ndarray) -> bool:
        """Return True if the target's HP bar has any red fill (>1%)."""
        return ScenarioRunner._target_hp_ratio(frame) > 0.01

    @staticmethod
    def _target_is_dead(frame: np.ndarray, matcher: TemplateMatcher) -> bool:
        """Target is visible but its HP bar is empty → dead."""
        return ScenarioRunner._has_target(frame, matcher) and not ScenarioRunner._target_has_hp(frame)

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
            log.debug("[patrol] patrol_zone template not found on map")
            return None

        px, py, conf = hit
        # Map centre in screen coords
        map_cx = rx + rw // 2
        map_cy = ry + rh // 2
        dx = px - map_cx
        dy = py - map_cy
        dist = math.hypot(dx, dy)
        log.debug(
            f"[patrol] zone offset dx={dx:.0f} dy={dy:.0f} "
            f"dist={dist:.0f}px  conf={conf:.2f}"
        )
        return dx, dy, dist

    def _walk_in_direction(
        self, ih: InputHandler, angle_rad: float, steps: int = 15,
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
                log.debug("[patrol] mob appeared nearby, stopping walk-back")
                break
            if i % 5 == 4:
                log.debug(f"[patrol] walking back… step {i+1}/{steps}")

    def _rotate_camera_toward_mob(
        self, ih: InputHandler, mob_dist: Optional[float] = None,
    ) -> bool:
        """Self-calibrating camera rotation (no feedback loop).

        1. Read mob angle on minimap
        2. Small test drag → measure how much the angle changed
        3. Calculate exact correction drag → apply
        4. One fine-tune pass if still off

        *mob_dist*: normalised distance of the target mob from the
        minimap centre, used to re-identify the same mob after the
        camera moves (avoids confusing multiple mobs).
        """
        capture = ScreenCapture()
        test_px = 25               # small known drag for calibration

        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2

        # ── Read 1: current mob position ──────────────────────
        frame = capture.grab()
        r1 = self._find_nearest_mob_on_minimap(frame, target_dist=mob_dist)
        if r1 is None:
            log.debug("[camera] no mob found")
            return False
        dx1, dy1, dist1 = r1
        angle1 = math.atan2(dx1, -dy1)   # 0 = north, + = clockwise
        log.debug(
            f"[camera] read1: dx={dx1:.3f} dy={dy1:.3f} "
            f"angle={math.degrees(angle1):.1f}°"
        )

        if abs(angle1) < 0.12:           # ~7° — close enough
            log.debug("[camera] mob already north")
            return True

        # ── Calibration: small test drag to the right ─────────
        ih.drag_to(cx, cy, cx + test_px, cy, duration=0.15, button="right")
        sleep(0.35)

        # ── Read 2: same mob after test drag ──────────────────
        frame = capture.grab()
        r2 = self._find_nearest_mob_on_minimap(frame, target_dist=dist1)
        if r2 is None:
            log.debug("[camera] lost mob after test drag")
            return False
        dx2, dy2, dist2 = r2
        angle2 = math.atan2(dx2, -dy2)

        # Normalise delta to [-π, π]
        delta = (angle2 - angle1 + math.pi) % (2 * math.pi) - math.pi
        log.debug(
            f"[camera] read2: dx={dx2:.3f} dy={dy2:.3f} "
            f"angle={math.degrees(angle2):.1f}° "
            f"delta={math.degrees(delta):.1f}°"
        )

        if abs(delta) < 0.02:            # ~1° — drag had no visible effect
            log.debug("[camera] test drag had no effect, big swing")
            big = 150 if angle1 > 0 else -150
            ih.drag_to(cx, cy, cx + big, cy, duration=0.4, button="right")
            sleep(0.35)
            return True

        # ── Calculate & apply main correction ─────────────────
        px_per_rad = test_px / delta      # px we must drag per radian
        correction = int(-angle2 * px_per_rad)
        correction = max(-500, min(500, correction))
        log.debug(
            f"[camera] px/rad={px_per_rad:.1f} correction={correction}px"
        )
        ih.drag_to(cx, cy, cx + correction, cy, duration=0.3, button="right")
        sleep(0.35)

        # ── Fine-tune if still off ────────────────────────────
        frame = capture.grab()
        r3 = self._find_nearest_mob_on_minimap(frame, target_dist=dist1)
        if r3 is not None:
            dx3, dy3, _ = r3
            angle3 = math.atan2(dx3, -dy3)
            log.debug(
                f"[camera] verify: dx={dx3:.3f} dy={dy3:.3f} "
                f"angle={math.degrees(angle3):.1f}°"
            )
            if abs(angle3) > 0.20:       # still off by >12°
                fine = int(-angle3 * px_per_rad * 0.7)
                fine = max(-300, min(300, fine))
                log.debug(f"[camera] fine-tune {fine}px")
                ih.drag_to(cx, cy, cx + fine, cy, duration=0.2, button="right")
                sleep(0.25)

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


    def _move_to_closest_mob(self, ih: InputHandler) -> None:
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

        # ── Phase 1: orient ─────────────────────────────────────
        if self._pre_oriented:
            log.debug("[move] pre-oriented – skipping orient phase")
            self._pre_oriented = False

            frame = capture.grab()
            result = self._find_nearest_mob_on_minimap(frame)
            if result is None:
                log.debug("[move] no mobs on minimap")
                return

            dx, dy, dist = result
            if dist <= close_range:
                log.debug("[move] mob already in range")
                return

            log.debug(f"[move] mob dir=({dx:.2f},{dy:.2f}) dist={dist:.2f}")

            # Rotate camera until mob is directly north
            if not self._rotate_camera_toward_mob(ih, mob_dist=dist):
                log.debug("[move] lost mob while rotating, aborting")
                return

        max_clicks = 40
        for i in range(max_clicks):
            ih.click(walk_x, walk_y)
            sleep(0.45)

            # Check distance only — no camera changes
            frame = capture.grab()
            result = self._find_nearest_mob_on_minimap(frame)
            if result is None:
                log.debug("[move] lost mob on minimap, stopping")
                break

            _, _, dist = result
            if dist <= close_range:
                log.debug("[move] mob is within range, done")
                break

            if i % 10 == 9:
                log.debug(f"[move] still walking… dist={dist:.2f}")

    async def move_to_mobs_and_attack_if_no_target(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        if self.busy:
            return
        if not self._has_target(frame, matcher):
            self.busy = True
            try:
                log.debug("[move_to_mobs_and_attack_if_no_target] no target – looking for mobs")
                self._move_to_closest_mob(ih)
                ih.press(config.KEY_NEXT_TARGET)
                sleep(0.2)
                ih.press(config.KEY_ATTACK)
            finally:
                self.busy = False

    async def assist_ppl_then_attack_on_dead_or_non_existing_target(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        if self.busy:
            return

        if not self._has_target(frame, matcher) or self._target_is_dead(frame, matcher):
            self.busy = True
            try:
                log.debug("[assist_ppl] no target – assisting ppl")
                ih.press(config.KEY_TARGET_PPL)
                ih.press(config.KEY_ASSIST)
                ih.press(config.KEY_ATTACK)
            finally:
                self.busy = False

    async def loot_on_dead_target(
        self,
        frame: np.ndarray,
        matcher: TemplateMatcher,
        ih: InputHandler,
        bus: EventBus,
        event: Optional[dict] = None,
    ) -> None:
        if self.busy:
            return
        has_target = self._has_target(frame, matcher)
        is_dead = self._target_is_dead(frame, matcher)

        if has_target and is_dead:
            self.busy = True
            log.debug("[loot_on_dead_target] target is dead – looting")
            try:
                for _ in range(5):
                    ih.press(config.KEY_LOOT)
                    sleep(0.1)
            finally:
                self._cancel_target(ih)
                self.busy = False

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
        if self.busy:
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

        self.busy = True
        try:
            log.debug(
                f"[pre_orient] target HP low ({ratio:.3f}), "
                f"pre-orienting toward next mob"
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
                log.debug("[pre_orient] no next mob found on minimap")
                return

            dx, dy, dist = result
            log.debug(
                f"[pre_orient] next mob dir=({dx:.2f},{dy:.2f}) "
                f"dist={dist:.2f}"
            )

            if self._rotate_camera_toward_mob(ih, mob_dist=dist):
                self._pre_oriented = True
                log.debug("[pre_orient] camera pre-oriented successfully")
            else:
                log.debug("[pre_orient] lost next mob while rotating")
        finally:
            self.busy = False

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
        if self.busy:
            return
        # Don't check while we have a live target
        if self._has_target(frame, matcher):
            return

        interval = getattr(config, "PATROL_CHECK_INTERVAL", 30.0)
        now = _time.monotonic()
        if now - self._last_patrol_check < interval:
            return

        self.busy = True
        try:
            self._last_patrol_check = now
            result = self._check_patrol_zone(ih, matcher)
            if result is None:
                log.debug("[patrol] could not locate zone – skipping")
                return

            dx, dy, dist = result
            threshold = getattr(config, "PATROL_MAX_DRIFT_PX", 60)
            if dist <= threshold:
                log.debug(f"[patrol] inside zone (drift={dist:.0f}px)")
                self._returning_to_zone = False
                return

            log.info(
                f"[patrol] outside zone (drift={dist:.0f}px > {threshold}px) "
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
                f"[patrol] need to face {math.degrees(target_angle):.1f}° "
                f"to reach zone"
            )

            # Use self-calibrating rotation via a virtual "mob" at that angle
            # on the minimap.  We place a synthetic reading and rotate toward it.
            # Simplest approach: just rotate camera by target_angle using
            # the same calibration technique.
            self._rotate_camera_by_angle(ih, target_angle)

            # Walk forward toward the zone
            steps = getattr(config, "PATROL_RETURN_STEPS", 15)
            self._walk_in_direction(ih, 0.0, steps=steps)  # 0 = straight ahead

        finally:
            self.busy = False

    def _rotate_camera_by_angle(
        self, ih: InputHandler, target_angle: float,
    ) -> None:
        """Rotate camera by *target_angle* radians using self-calibration.

        Uses a small test drag to measure px-per-radian, then applies
        the full rotation.  Works without needing a mob on the minimap.
        """
        if abs(target_angle) < 0.12:  # < ~7°, not worth rotating
            return

        test_px = 25
        with mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
        cx = mon["left"] + mon["width"] // 2
        cy = mon["top"] + mon["height"] // 2

        capture = ScreenCapture()

        # ── Read 1: pick any mob (or just a reference point) ──
        frame1 = capture.grab()
        r1 = self._find_nearest_mob_on_minimap(frame1)

        # Test drag right
        ih.drag_to(cx, cy, cx + test_px, cy, duration=0.15, button="right")
        sleep(0.35)

        frame2 = capture.grab()
        r2 = self._find_nearest_mob_on_minimap(frame2)

        if r1 is not None and r2 is not None:
            a1 = math.atan2(r1[0], -r1[1])
            a2 = math.atan2(r2[0], -r2[1])
            delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
            if abs(delta) > 0.02:
                px_per_rad = test_px / delta
                correction = int(target_angle * px_per_rad)
                correction = max(-500, min(500, correction))
                log.debug(
                    f"[patrol_rotate] calibrated px/rad={px_per_rad:.1f}, "
                    f"correction={correction}px"
                )
                ih.drag_to(cx, cy, cx + correction, cy, duration=0.3, button="right")
                sleep(0.3)
                return

        # Fallback: rough estimate (~100px per 90°)
        rough_px = int(target_angle / (math.pi / 2) * 100)
        rough_px = max(-400, min(400, rough_px))
        log.debug(f"[patrol_rotate] fallback drag {rough_px}px")
        ih.drag_to(cx, cy, cx + rough_px, cy, duration=0.3, button="right")
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
            log.debug("[auto_attack] no target – searching")
            ih.press("f1")
        else:
            attack_btn = matcher.find(frame, "attack_button")
            if ih.click_template(attack_btn):
                log.debug("[auto_attack] attacking")

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
            log.info(f"[loot_nearby] loot at ({x},{y}) conf={conf:.2f}")
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
            log.info(f"[react_remote_loot] partner found loot: {event}")

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
            log.warning("[stop_if_exit_game] exit menu detected – stopping bot")
            raise BotStopRequested("exit_game detected")

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
            "loot_on_dead_target": self.loot_on_dead_target,
            "pre_orient_to_next_mob": self.pre_orient_to_next_mob,
            "return_to_patrol_zone": self.return_to_patrol_zone,
            "move_to_mobs_and_attack_if_no_target": self.move_to_mobs_and_attack_if_no_target,
            "stop_if_exit_game": self.stop_if_exit_game,
        }
        return [registry[n] for n in names if n in registry]

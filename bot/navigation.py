"""
bot/navigation.py – camera rotation, character movement, and patrol helpers

All functions are *stateless* with respect to game state; they accept
an ``InputHandler`` for sending inputs and an optional ``check_exit``
callback so the caller can inject its own exit-detection logic.
"""

from __future__ import annotations

import logging
import math
import time as _time
from time import sleep
from typing import Optional, Callable

import config
from bot.screen import ScreenCapture, TemplateMatcher
from bot.input_handler import InputHandler
from bot import vision

log = logging.getLogger(__name__)

# Type alias for the check-exit callback.
CheckExitFn = Optional[Callable[[], None]]


# ── Utilities ────────────────────────────────────────────────────

def get_screen_center(capture: Optional[ScreenCapture] = None) -> tuple[int, int]:
    """Return ``(x, y)`` of the primary monitor's centre (screen coords)."""
    if capture is not None:
        mon = capture.monitor
    else:
        import mss as _mss
        with _mss.mss() as sct:
            mon = sct.monitors[config.MONITOR_INDEX]
    return mon["left"] + mon["width"] // 2, mon["top"] + mon["height"] // 2


# ── Camera rotation ──────────────────────────────────────────────

def rotate_camera_toward_mob(
    ih: InputHandler,
    mob_dist: Optional[float] = None,
    check_exit: CheckExitFn = None,
    capture: Optional[ScreenCapture] = None,
) -> bool:
    """Rotate camera until a mob is directly north on the minimap.

    Holds right-mouse-button and moves the mouse in small steps.
    This avoids the overhead of full press/move/release per nudge.

    Logic:
      1. If any mob is already in the north cone → done.
      2. Pick a direction based on nearest mob, commit to it.
      3. Hold right-click, nudge mouse step-by-step until a mob is north.

    Returns ``True`` if a mob ended up north, ``False`` if all mobs lost.
    """
    _cap = capture or ScreenCapture()
    cx, cy = get_screen_center(_cap)

    north_cone_deg = getattr(config, "CAMERA_NORTH_THRESHOLD_DEG", 20)
    step_px        = getattr(config, "CAMERA_STEP_PX", 1)
    max_passes     = getattr(config, "CAMERA_MAX_PASSES", 180)
    settle_ms      = getattr(config, "CAMERA_SETTLE_MS", 30) / 1000.0

    # ── 1. Already have a mob at north? ──────────────────────────
    frame = _cap.grab()
    if vision.has_mob_at_north(frame, half_cone_deg=north_cone_deg):
        log.debug("[camera:rotate] mob already at north – no rotation needed")
        return True

    # ── 2. Pick a committed direction ────────────────────────────
    r = vision.find_nearest_mob_on_minimap(frame, target_dist=mob_dist)
    if r is None:
        log.debug("[camera:rotate] no mobs on minimap – cannot orient")
        return False

    dx, dy, dist = r
    angle = math.atan2(dx, -dy)                     # +ve = mob is to the right
    direction = 1 if angle > 0 else -1               # commit: never change
    log.debug(
        f"[camera:rotate] mob at {math.degrees(angle):+.1f}° – "
        f"will nudge {'right' if direction > 0 else 'left'}"
    )

    # ── 3. Hold right-click and nudge ───────────────────────────
    ih.move_to(cx, cy)
    sleep(0.02)
    ih.mouse_down("right")
    sleep(0.02)

    current_x = cx
    success = False
    try:
        for i in range(max_passes):
            current_x += direction * step_px
            ih.move_to(current_x, cy)
            sleep(settle_ms)

            frame = _cap.grab()

            if vision.has_mob_at_north(frame, half_cone_deg=north_cone_deg):
                log.debug(
                    f"[camera:rotate] mob entered north cone after {i+1} nudges"
                )
                success = True
                break

            # If ALL mobs disappeared, bail out
            if vision.find_nearest_mob_on_minimap(frame) is None:
                log.debug(f"[camera:rotate] all mobs lost after {i+1} nudges")
                break

            if check_exit:
                check_exit()

        if not success:
            log.debug(f"[camera:rotate] max {max_passes} nudges reached")
    finally:
        ih.mouse_up("right")
        sleep(0.02)

    return success


def rotate_camera_by_angle(
    ih: InputHandler,
    target_angle: float,
    capture: Optional[ScreenCapture] = None,
) -> None:
    """Rotate camera by *target_angle* radians using the calibration constant."""
    if abs(target_angle) < 0.12:   # < ~7°, not worth rotating
        return

    px_per_rad = getattr(config, "CAMERA_PX_PER_RAD", 19.0)
    _cap = capture or ScreenCapture()
    cx, cy = get_screen_center(_cap)

    correction = int(target_angle * px_per_rad)
    correction = max(-300, min(300, correction))
    log.debug(
        f"[patrol:rotate] angle={math.degrees(target_angle):.1f}° "
        f"correction={correction}px"
    )
    ih.drag_to(cx, cy, cx + correction, cy, duration=0.25, button="right")
    sleep(0.3)


# ── Walking / movement ──────────────────────────────────────────

def walk_in_direction(
    ih: InputHandler,
    angle_rad: float,
    steps: int = 15,
    check_exit: CheckExitFn = None,
    capture: Optional[ScreenCapture] = None,
) -> None:
    """Click-walk in *angle_rad* direction (0 = north, positive = clockwise)."""
    walk_radius = getattr(config, "MOVE_FORWARD_CLICK_PX", 250)
    _cap = capture or ScreenCapture()
    scr_cx, scr_cy = get_screen_center(_cap)

    click_x = scr_cx + int(walk_radius * math.sin(angle_rad))
    click_y = scr_cy - int(walk_radius * math.cos(angle_rad))

    close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
    for i in range(steps):
        ih.click(click_x, click_y)
        sleep(0.45)

        frame = _cap.grab()
        result = vision.find_nearest_mob_on_minimap(frame)
        if result is not None and result[2] <= close_range:
            log.debug(
                f"[patrol:walk] mob appeared nearby (dist={result[2]:.2f}), "
                f"aborting walk at step {i+1}/{steps}"
            )
            break

        if check_exit:
            check_exit()
        if i % 5 == 4:
            log.debug(
                f"[patrol:walk] step {i+1}/{steps} – no mobs nearby yet"
            )


def move_to_closest_mob(
    ih: InputHandler,
    pre_oriented: bool = False,
    check_exit: CheckExitFn = None,
    capture: Optional[ScreenCapture] = None,
) -> None:
    """Orient the camera toward the nearest mob and walk to it.

    *pre_oriented*: skip the orientation phase (caller already rotated).
    """
    close_range = getattr(config, "MOB_CLOSE_RANGE", 0.15)
    forward_px = getattr(config, "MOVE_FORWARD_CLICK_PX", 250)
    _cap = capture or ScreenCapture()
    scr_cx, scr_cy = get_screen_center(_cap)
    walk_x = scr_cx
    walk_y = scr_cy - forward_px

    _move_t0 = _time.monotonic()

    if pre_oriented:
        log.debug(
            "[move:orient] skipping – already pre-oriented from "
            "pre_orient_to_next_mob"
        )
    else:
        frame = _cap.grab()
        result = vision.find_nearest_mob_on_minimap(frame)
        if result is None:
            log.debug("[move:orient] no mobs detected on minimap – aborting")
            return
        dx, dy, dist = result
        if dist <= close_range:
            log.debug(
                f"[move:orient] mob already in range "
                f"(dist={dist:.2f} <= {close_range})"
            )
            return
        log.debug(
            f"[move:orient] nearest mob at dir=({dx:.2f},{dy:.2f}) "
            f"dist={dist:.2f} – rotating camera"
        )
        if not rotate_camera_toward_mob(ih, mob_dist=dist, check_exit=check_exit, capture=_cap):
            log.debug(
                f"[move:orient] mob lost during camera rotation – aborting "
                f"({(_time.monotonic()-_move_t0)*1000:.0f}ms)"
            )
            return

    log.debug(
        f"[move:orient] orient phase done "
        f"({(_time.monotonic()-_move_t0)*1000:.0f}ms) – starting walk"
    )

    _walk_t0 = _time.monotonic()
    max_clicks = 40
    for i in range(max_clicks):
        ih.click(walk_x, walk_y)
        sleep(0.45)

        frame = _cap.grab()
        result = vision.find_nearest_mob_on_minimap(frame)
        if result is None:
            log.debug(
                f"[move:walk] mob lost on minimap at step {i+1}/{max_clicks} "
                f"({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)"
            )
            break
        _, _, dist = result
        if dist <= close_range:
            log.debug(
                f"[move:walk] mob in range (dist={dist:.2f}) after {i+1} steps "
                f"({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)"
            )
            break
        if i % 5 == 4:
            log.debug(
                f"[move:walk] step {i+1}/{max_clicks} – dist={dist:.2f} "
                f"({(_time.monotonic()-_walk_t0)*1000:.0f}ms walk)"
            )
        if check_exit:
            check_exit()

    log.debug(
        f"[move:total] move_to_closest_mob finished "
        f"({(_time.monotonic()-_move_t0)*1000:.0f}ms total)"
    )


# ── Patrol zone ──────────────────────────────────────────────────

def check_patrol_zone(
    ih: InputHandler,
    matcher: TemplateMatcher,
    capture: Optional[ScreenCapture] = None,
) -> Optional[tuple[float, float, float]]:
    """Open map, find the ``patrol_zone`` template, return offset or ``None``.

    Returns ``(dx, dy, dist)`` pixel offset from map centre, or ``None``
    if the template was not found.
    """
    _cap = capture or ScreenCapture()
    ih.hotkey("alt", "m")
    sleep(0.6)

    frame = _cap.grab()
    rx, ry, rw, rh = config.REGION_MAP
    hit = matcher.find(frame, "patrol_zone", region=config.REGION_MAP)

    ih.hotkey("alt", "m")
    sleep(0.3)

    if hit is None:
        log.debug("[patrol:map] patrol_zone template not found on map")
        return None

    px, py, conf = hit
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

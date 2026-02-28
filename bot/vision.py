"""
bot/vision.py – stateless computer-vision helpers for frame analysis

All functions are pure: they take a frame (and optionally a matcher /
config values) and return a result.  No game state is modified.
"""

from __future__ import annotations

import logging
import math
import time as _time
from time import sleep
from typing import Optional

import cv2
import numpy as np

import config
from bot.screen import ScreenCapture, TemplateMatcher

log = logging.getLogger(__name__)

# ── Per-frame cache for target_hp_ratio ──────────────────────────
# Avoids redundant computation when multiple scenarios call it on the
# same captured frame within a single tick.
_hp_cache_frame_id: int = 0
_hp_cache_value: float = 0.0


# ─────────────────────────────────────────────────────────────────
#  Target helpers
# ─────────────────────────────────────────────────────────────────

def has_target(frame: np.ndarray, matcher: TemplateMatcher) -> bool:
    """Return True if the target frame is visible (template-based)."""
    return matcher.find(frame, "has_target", region=config.REGION_TARGET) is not None


def target_hp_ratio(frame: np.ndarray) -> float:
    """Return the red fill ratio (0.0–1.0) of the target HP bar.

    Counts bright-red pixels in the HP-bar strip using HSV thresholds.
    Results are cached per frame ``id()`` so multiple callers in the
    same tick don't recompute or double-log.
    """
    global _hp_cache_frame_id, _hp_cache_value
    fid = id(frame)
    if _hp_cache_frame_id == fid:
        return _hp_cache_value

    rx, ry, rw, rh = config.REGION_TARGET_HP_BAR
    roi = frame[ry:ry + rh, rx:rx + rw]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Red wraps around H=0/180 in OpenCV's 0-180 hue range
    lo1 = np.array([0, 120, 100], dtype=np.uint8)
    hi1 = np.array([10, 255, 255], dtype=np.uint8)
    lo2 = np.array([165, 120, 100], dtype=np.uint8)
    hi2 = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    red_pixels = np.count_nonzero(mask)
    total = mask.size
    ratio = red_pixels / total if total else 0.0
    log.debug(f"[target:hp] red_fill={ratio:.3f} ({red_pixels}/{total} px)")

    _hp_cache_frame_id = fid
    _hp_cache_value = ratio
    return ratio


def target_has_hp(frame: np.ndarray) -> bool:
    """Return True if the target's HP bar has any red fill (>1 %)."""
    return target_hp_ratio(frame) > 0.01


def target_is_dead(
    frame: np.ndarray,
    matcher: TemplateMatcher,
    capture: Optional[ScreenCapture] = None,
) -> bool:
    """Target is visible but its HP bar is empty → dead.

    Uses a brief confirmation re-check to avoid false positives when
    a sliver of HP remains.
    """
    if not has_target(frame, matcher):
        return False
    if target_has_hp(frame):
        return False

    _t0 = _time.monotonic()
    sleep(0.15)
    _cap = capture or ScreenCapture()
    frame2 = _cap.grab()
    if not has_target(frame2, matcher):
        log.debug(
            f"[target:dead_check] target vanished during confirm "
            f"({(_time.monotonic() - _t0) * 1000:.0f}ms)"
        )
        return False
    is_dead = not target_has_hp(frame2)
    log.debug(
        f"[target:dead_check] confirmed={'DEAD' if is_dead else 'ALIVE'} "
        f"({(_time.monotonic() - _t0) * 1000:.0f}ms)"
    )
    return is_dead


# ─────────────────────────────────────────────────────────────────
#  Exit-menu detection
# ─────────────────────────────────────────────────────────────────

def is_exit_visible(frame: np.ndarray, matcher: TemplateMatcher) -> bool:
    """Return True if the in-game exit/quit menu is visible."""
    return matcher.find(frame, "exit_game", region=config.REGION_GENERAL_MENU) is not None


# ─────────────────────────────────────────────────────────────────
#  Minimap mob detection
# ─────────────────────────────────────────────────────────────────

def find_nearest_mob_on_minimap(
    frame: np.ndarray,
    target_dist: Optional[float] = None,
    min_dist: Optional[float] = None,
) -> Optional[tuple[float, float, float]]:
    """Detect red dots (mobs) on the circular minimap.

    Returns ``(dx, dy, dist)`` where *dx*/*dy* is the direction vector
    normalised to [-1, 1] and *dist* is the normalised distance
    (0 = centre, 1 = edge).  Returns ``None`` if no mobs found.

    Parameters
    ----------
    target_dist : float, optional
        Prefer the mob whose normalised distance from the centre best
        matches this value (used to track the same mob across camera
        rotations).
    min_dist : float, optional
        Ignore mobs closer than this normalised distance (e.g. the mob
        we are currently fighting).
    """
    rx, ry, rw, rh = config.REGION_MINIMAP
    minimap = frame[ry:ry + rh, rx:rx + rw]

    # Circular mask – the minimap is round
    h, w = minimap.shape[:2]
    center_x, center_y = w // 2, h // 2
    radius = min(center_x, center_y)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # Detect red pixels in HSV (red wraps around H=0/180)
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    lo1 = np.array([0, 100, 80], dtype=np.uint8)
    hi1 = np.array([10, 255, 255], dtype=np.uint8)
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
        # Skip the very centre (that's the player, not a mob)
        if dist_px < 5:
            continue
        norm_dx = cdx / radius
        norm_dy = cdy / radius
        norm_dist = math.hypot(norm_dx, norm_dy)

        if min_dist is not None and norm_dist < min_dist:
            continue

        if target_dist is not None:
            score = abs(norm_dist - target_dist)
        else:
            score = dist_px

        if score < best_score:
            best_score = score
            best_dx, best_dy = norm_dx, norm_dy
            best_norm_dist = norm_dist

    if best_score == float("inf"):
        return None
    return best_dx, best_dy, best_norm_dist

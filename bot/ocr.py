"""
bot/ocr.py – OCR-based character-stat reading utilities

Provides atomic, reusable functions for:
  • Image preprocessing (white-text isolation, upscaling)
  • Running easyocr
  • Parsing individual stat fields (current/max pairs, level, XP %)
  • A top-level ``read_stats()`` that orchestrates the full pipeline
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, fields
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

# ── Lazy-loaded easyocr reader (heavy import, created once) ──────
_ocr_reader = None


def get_ocr_reader():
    """Return a shared easyocr ``Reader`` instance (created on first call)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


# ── Data container ───────────────────────────────────────────────

@dataclass
class StatsReading:
    """Parsed character stats from a single OCR pass.

    Fields default to ``0`` / ``0.0`` which means "not detected".
    The caller should fall back to previously-known values for any
    field that is still at its default.
    """
    level: int = 0
    cp_current: int = 0
    cp_max: int = 0
    hp_current: int = 0
    hp_max: int = 0
    mp_current: int = 0
    mp_max: int = 0
    xp_percent: float = 0.0


# ── Preprocessing ────────────────────────────────────────────────

def preprocess_stats_roi(frame: np.ndarray, scale: int = 3) -> np.ndarray:
    """Extract and preprocess the stats region for OCR.

    Isolates white/bright text (numbers on the coloured bars),
    upscales for accuracy, and pads for OCR edge handling.

    Returns a single-channel padded binary image.
    """
    rx, ry, rw, rh = config.REGION_GENERAL_STATS
    roi = frame[ry:ry + rh, rx:rx + rw]

    roi_big = cv2.resize(
        roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC,
    )

    b, g, r_ch = cv2.split(roi_big)
    white_mask = cv2.bitwise_and(
        cv2.bitwise_and(
            cv2.threshold(r_ch, 160, 255, cv2.THRESH_BINARY)[1],
            cv2.threshold(g, 160, 255, cv2.THRESH_BINARY)[1],
        ),
        cv2.threshold(b, 160, 255, cv2.THRESH_BINARY)[1],
    )
    return cv2.copyMakeBorder(
        white_mask, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0,
    )


# ── OCR execution ────────────────────────────────────────────────

def run_ocr(image: np.ndarray) -> list[str]:
    """Run easyocr on a preprocessed binary image.

    Returns a list of whitespace-stripped text tokens (may be empty).
    """
    try:
        reader = get_ocr_reader()
        results = reader.readtext(
            image,
            allowlist="0123456789/.%CPHMXcphmpx ",
            detail=0,
            paragraph=False,
        )
        return [s.strip() for s in results if s.strip()]
    except Exception as e:
        log.debug(f"[stats:error] OCR failed: {e}")
        return []


# ── Atomic parsers ───────────────────────────────────────────────

def parse_stat_pair(text: str) -> Optional[tuple[int, int]]:
    """Parse ``'current/max'`` or concatenated digits from OCR text.

    Examples::

        '1274/1274'  →  (1274, 1274)
        '1274 /1274' →  (1274, 1274)
        '4401440'    →  (440, 1440)   # even-split fallback
    """
    clean = re.sub(r"[^0-9/]", "", text)
    m = re.search(r"(\d+)/(\d+)", clean)
    if m:
        return int(m.group(1)), int(m.group(2))

    digits = re.sub(r"\D", "", clean)
    if len(digits) >= 2:
        if len(digits) % 2 == 0:
            half = len(digits) // 2
            return int(digits[:half]), int(digits[half:])
        mid = len(digits) // 2
        left, right = digits[:mid], digits[mid + 1:]
        if left and right and left == right:
            return int(left), int(right)
        left2, right2 = digits[: mid + 1], digits[mid + 1:]
        if left2 and right2:
            return int(left2), int(right2)
    return None


def find_label_value(
    raw_lines: list[str],
    upper_lines: list[str],
    label: str,
) -> Optional[str]:
    """Find *label* (e.g. ``'CP'``) in OCR results and return the
    associated value text (the next token, or the tail of the same
    token if the label is embedded).
    """
    for i, item in enumerate(upper_lines):
        if item == label and i + 1 < len(upper_lines):
            return raw_lines[i + 1]
        if item.startswith(label) and len(item) > len(label):
            return item[len(label):]
    return None


def parse_level(raw_lines: list[str], upper_lines: list[str]) -> Optional[int]:
    """Extract character level (small standalone number before first stat label)."""
    first_label_idx = _first_label_index(upper_lines)
    for i in range(first_label_idx):
        lm = re.search(r"(\d{1,3})", raw_lines[i])
        if lm:
            val = int(lm.group(1))
            if 1 <= val <= 99:
                return val
    return None


def parse_xp_percent(
    raw_lines: list[str],
    upper_lines: list[str],
) -> Optional[float]:
    """Extract XP percentage from the tail of OCR tokens.

    Handles both ``'26.36%'`` and split tokens like ``'26'`` + ``'36%'``.
    """
    first_label_idx = _first_label_index(upper_lines)
    tail = " ".join(raw_lines[first_label_idx:])

    xp_m = re.search(r"(\d+\.\d+)\s*%", tail)
    if xp_m:
        return float(xp_m.group(1))

    xp_parts = re.findall(r"(\d+)\s*%", tail)
    xp_digits = re.findall(r"\b(\d{1,2})\b", tail)
    if xp_parts and len(xp_digits) >= 2:
        try:
            return float(f"{xp_digits[-2]}.{xp_parts[-1]}")
        except (ValueError, IndexError):
            pass
    return None


# ── Internal helpers ─────────────────────────────────────────────

def _first_label_index(upper_lines: list[str]) -> int:
    """Return the index of the first CP/HP/MP label in *upper_lines*."""
    for i, item in enumerate(upper_lines):
        if item in ("CP", "HP", "MP"):
            return i
    return len(upper_lines)


# ── Top-level pipeline ───────────────────────────────────────────

def read_stats(frame: np.ndarray) -> Optional[StatsReading]:
    """Full OCR pipeline: preprocess → read → parse all stats.

    Returns a :class:`StatsReading` with whatever fields could be
    parsed, or ``None`` if OCR produced no usable tokens.
    """
    padded = preprocess_stats_roi(frame)
    raw_lines = run_ocr(padded)
    if not raw_lines:
        return None

    log.debug(f"[stats:ocr] raw tokens: {raw_lines}")

    upper_lines = [s.upper().strip() for s in raw_lines]
    reading = StatsReading()

    # CP / HP / MP
    for label, attr_cur, attr_max in [
        ("CP", "cp_current", "cp_max"),
        ("HP", "hp_current", "hp_max"),
        ("MP", "mp_current", "mp_max"),
    ]:
        text = find_label_value(raw_lines, upper_lines, label)
        if text:
            pair = parse_stat_pair(text)
            if pair:
                setattr(reading, attr_cur, pair[0])
                setattr(reading, attr_max, pair[1])

    # Level
    level = parse_level(raw_lines, upper_lines)
    if level is not None:
        reading.level = level

    # XP
    xp = parse_xp_percent(raw_lines, upper_lines)
    if xp is not None:
        reading.xp_percent = xp

    return reading

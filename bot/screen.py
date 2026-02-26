"""
bot/screen.py – screenshot capture and image analysis
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

import config


class ScreenCapture:
    """Captures the game monitor at a configured FPS."""

    def __init__(self) -> None:
        self._sct = mss.mss()
        self._monitor = self._sct.monitors[config.MONITOR_INDEX]

    @property
    def monitor(self) -> dict:
        return self._monitor

    def grab(self) -> np.ndarray:
        """Return current monitor content as a BGR numpy array."""
        raw = self._sct.grab(self._monitor)
        frame = np.array(raw)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def save_screenshot(self, path: str | Path) -> None:
        cv2.imwrite(str(path), self.grab())


class TemplateMatcher:
    """
    Loads PNG templates from TEMPLATES_DIR and looks for them
    in a given frame using normalised cross-correlation.
    """

    def __init__(self) -> None:
        self._templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        base = Path(config.TEMPLATES_DIR)
        if not base.exists():
            return
        for path in base.glob("*.png"):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                self._templates[path.stem] = img
        print(f"[TemplateMatcher] loaded {len(self._templates)} templates")

    def reload(self) -> None:
        self._templates.clear()
        self._load_templates()

    def find(
        self,
        frame: np.ndarray,
        template_name: str,
        threshold: float = config.MATCH_THRESHOLD,
    ) -> Optional[tuple[int, int, float]]:
        """
        Search for a named template in *frame*.

        Returns (center_x, center_y, confidence) if found, else None.
        """
        tpl = self._templates.get(template_name)
        if tpl is None:
            return None

        result = cv2.matchTemplate(frame, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            h, w = tpl.shape[:2]
            cx = max_loc[0] + w // 2
            cy = max_loc[1] + h // 2
            return cx, cy, float(max_val)
        return None

    def find_all(
        self,
        frame: np.ndarray,
        threshold: float = config.MATCH_THRESHOLD,
    ) -> dict[str, tuple[int, int, float]]:
        """Return every template that is currently visible on screen."""
        found = {}
        for name in self._templates:
            hit = self.find(frame, name, threshold)
            if hit:
                found[name] = hit
        return found

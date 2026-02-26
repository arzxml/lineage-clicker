"""
bot/input_handler.py – mouse and keyboard actions
"""

from __future__ import annotations

import time

import pyautogui

# Safety setting: move mouse to corner to abort
pyautogui.FAILSAFE = True
# Pause between pyautogui calls (seconds)
pyautogui.PAUSE = 0.05


class InputHandler:
    """Wraps pyautogui to provide click / key-press helpers."""

    # ── Mouse ─────────────────────────────────────────────────────────

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Move to (x, y) and click."""
        pyautogui.click(x, y, button=button, clicks=clicks)

    def double_click(self, x: int, y: int) -> None:
        pyautogui.doubleClick(x, y)

    def right_click(self, x: int, y: int) -> None:
        pyautogui.rightClick(x, y)

    def move_to(self, x: int, y: int, duration: float = 0.1) -> None:
        pyautogui.moveTo(x, y, duration=duration)

    def drag_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.3,
        button: str = "left",
    ) -> None:
        pyautogui.moveTo(x1, y1)
        pyautogui.dragTo(x2, y2, duration=duration, button=button)

    # ── Keyboard ──────────────────────────────────────────────────────

    def press(self, key: str) -> None:
        """Tap a single key (e.g. 'f1', 'enter', 'space')."""
        pyautogui.press(key)

    def hotkey(self, *keys: str) -> None:
        """Press a combination of keys (e.g. 'ctrl', 'a')."""
        pyautogui.hotkey(*keys)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        pyautogui.typewrite(text, interval=interval)

    def key_down(self, key: str) -> None:
        pyautogui.keyDown(key)

    def key_up(self, key: str) -> None:
        pyautogui.keyUp(key)

    # ── Utilities ─────────────────────────────────────────────────────

    def click_template(
        self,
        match: tuple[int, int, float] | None,
        button: str = "left",
    ) -> bool:
        """
        Convenience: pass a TemplateMatcher.find() result directly.
        Returns True if a click was performed.
        """
        if match is None:
            return False
        x, y, _ = match
        self.click(x, y, button=button)
        return True

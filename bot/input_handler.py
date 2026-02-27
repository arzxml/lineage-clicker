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


import ctypes
from ctypes import wintypes

MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP   = 0x0100
XBUTTON1 = 0x0001  # Mouse4 (back)
XBUTTON2 = 0x0002  # Mouse5 (forward)

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("mi", MOUSEINPUT)]

def click_mouse4():
    x_down = INPUT(type=0, mi=MOUSEINPUT(
        dx=0, dy=0, mouseData=XBUTTON1,
        dwFlags=MOUSEEVENTF_XDOWN, time=0, dwExtraInfo=None))
    x_up = INPUT(type=0, mi=MOUSEINPUT(
        dx=0, dy=0, mouseData=XBUTTON1,
        dwFlags=MOUSEEVENTF_XUP, time=0, dwExtraInfo=None))
    ctypes.windll.user32.SendInput(1, ctypes.byref(x_down), ctypes.sizeof(INPUT))
    ctypes.windll.user32.SendInput(1, ctypes.byref(x_up), ctypes.sizeof(INPUT))

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

"""
bot/interception_wrapper.py – kernel-level input via the Interception driver

Provides high-level click / key-press helpers that inject input at the
driver level, identical to physical hardware.  Games that ignore SendInput
(pyautogui / pydirectinput) will accept this.

Requirements
------------
1. Interception driver installed:
       https://github.com/oblitum/Interception/releases
       Run `install-interception.exe /install` as admin, then reboot.
2. pip install interception-python
"""

from __future__ import annotations

import ctypes
import time
from typing import Optional

from interception import ffi, lib

# ── Scan-code table (Set 1 / XT) ─────────────────────────────────────
# Extended keys (arrows, home, end, ins, del, numpad enter, right ctrl/alt,
# etc.) need the E0 prefix flag.
#
# Add more entries as needed – the key names mirror pyautogui conventions.

_SCAN: dict[str, tuple[int, bool]] = {
    # Row 0: Esc / F-keys
    "esc":          (0x01, False),
    "escape":       (0x01, False),
    "f1":           (0x3B, False),
    "f2":           (0x3C, False),
    "f3":           (0x3D, False),
    "f4":           (0x3E, False),
    "f5":           (0x3F, False),
    "f6":           (0x40, False),
    "f7":           (0x41, False),
    "f8":           (0x42, False),
    "f9":           (0x43, False),
    "f10":          (0x44, False),
    "f11":          (0x57, False),
    "f12":          (0x58, False),

    # Row 1: number row
    "`":            (0x29, False),
    "1":            (0x02, False),
    "2":            (0x03, False),
    "3":            (0x04, False),
    "4":            (0x05, False),
    "5":            (0x06, False),
    "6":            (0x07, False),
    "7":            (0x08, False),
    "8":            (0x09, False),
    "9":            (0x0A, False),
    "0":            (0x0B, False),
    "-":            (0x0C, False),
    "=":            (0x0D, False),
    "backspace":    (0x0E, False),

    # Row 2
    "tab":          (0x0F, False),
    "q":            (0x10, False),
    "w":            (0x11, False),
    "e":            (0x12, False),
    "r":            (0x13, False),
    "t":            (0x14, False),
    "y":            (0x15, False),
    "u":            (0x16, False),
    "i":            (0x17, False),
    "o":            (0x18, False),
    "p":            (0x19, False),
    "[":            (0x1A, False),
    "]":            (0x1B, False),
    "\\":           (0x2B, False),

    # Row 3
    "capslock":     (0x3A, False),
    "a":            (0x1E, False),
    "s":            (0x1F, False),
    "d":            (0x20, False),
    "f":            (0x21, False),
    "g":            (0x22, False),
    "h":            (0x23, False),
    "j":            (0x24, False),
    "k":            (0x25, False),
    "l":            (0x26, False),
    ";":            (0x27, False),
    "'":            (0x28, False),
    "enter":        (0x1C, False),
    "return":       (0x1C, False),

    # Row 4
    "shiftleft":    (0x2A, False),
    "shift":        (0x2A, False),
    "z":            (0x2C, False),
    "x":            (0x2D, False),
    "c":            (0x2E, False),
    "v":            (0x2F, False),
    "b":            (0x30, False),
    "n":            (0x31, False),
    "m":            (0x32, False),
    ",":            (0x33, False),
    ".":            (0x34, False),
    "/":            (0x35, False),
    "shiftright":   (0x36, False),

    # Row 5
    "ctrl":         (0x1D, False),
    "ctrlleft":     (0x1D, False),
    "win":          (0x5B, True),
    "winleft":      (0x5B, True),
    "alt":          (0x38, False),
    "altleft":      (0x38, False),
    "space":        (0x39, False),
    "altright":     (0x38, True),
    "winright":     (0x5C, True),
    "ctrlright":    (0x1D, True),

    # Navigation cluster (extended)
    "printscreen":  (0x37, True),
    "scrolllock":   (0x46, False),
    "pause":        (0x45, False),
    "insert":       (0x52, True),
    "home":         (0x47, True),
    "pageup":       (0x49, True),
    "delete":       (0x53, True),
    "end":          (0x4F, True),
    "pagedown":     (0x51, True),

    # Arrow keys (extended)
    "up":           (0x48, True),
    "left":         (0x4B, True),
    "down":         (0x50, True),
    "right":        (0x4D, True),

    # Numpad
    "numlock":      (0x45, False),
    "num0":         (0x52, False),
    "num1":         (0x4F, False),
    "num2":         (0x50, False),
    "num3":         (0x51, False),
    "num4":         (0x4B, False),
    "num5":         (0x4C, False),
    "num6":         (0x4D, False),
    "num7":         (0x47, False),
    "num8":         (0x48, False),
    "num9":         (0x49, False),
    "numpaddecimal":(0x53, False),
    "numpadenter":  (0x1C, True),
    "numpadplus":   (0x4E, False),
    "numpadminus":  (0x4A, False),
    "numpadmultiply":(0x37, False),
    "numpaddivide": (0x35, True),
}

# ── Mouse button state flags ─────────────────────────────────────────

_MOUSE_BTN = {
    "left":   (lib.INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
               lib.INTERCEPTION_MOUSE_LEFT_BUTTON_UP),
    "right":  (lib.INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN,
               lib.INTERCEPTION_MOUSE_RIGHT_BUTTON_UP),
    "middle": (lib.INTERCEPTION_MOUSE_BUTTON_3_DOWN,
               lib.INTERCEPTION_MOUSE_BUTTON_3_UP),
}

# ── Screen metrics (for absolute mouse positioning) ──────────────────
# Use the VIRTUAL desktop so multi-monitor setups work correctly.

_user32 = ctypes.windll.user32
_user32.SetProcessDPIAware()

# Virtual desktop spans ALL monitors
_VIRT_LEFT = _user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
_VIRT_TOP  = _user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
_VIRT_W    = _user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
_VIRT_H    = _user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN

# Interception absolute coords are 0‑65535 across the virtual desktop
_NORM_X = 65535.0 / _VIRT_W
_NORM_Y = 65535.0 / _VIRT_H


class Interception:
    """High-level wrapper around the Interception driver."""

    # Default device indices (keyboard=1, mouse=11 → first physical devices)
    def __init__(self, keyboard_device: int = 1, mouse_device: int = 11):
        self._ctx = lib.interception_create_context()
        if self._ctx == ffi.NULL:
            raise RuntimeError(
                "Could not create Interception context – is the driver installed?"
            )
        self._kbd = keyboard_device
        self._mouse = mouse_device

    # ── Cleanup ───────────────────────────────────────────────────────

    def destroy(self) -> None:
        if self._ctx != ffi.NULL:
            lib.interception_destroy_context(self._ctx)
            self._ctx = ffi.NULL

    def __del__(self):
        self.destroy()

    # ── Low-level helpers ─────────────────────────────────────────────

    def _send_key(self, scan: int, state: int) -> None:
        stroke = ffi.new("InterceptionKeyStroke *")
        stroke.code = scan
        stroke.state = state
        stroke.information = 0
        lib.interception_send(
            self._ctx,
            self._kbd,
            ffi.cast("InterceptionStroke *", stroke),
            1,
        )

    def _send_mouse(self, *, state: int = 0, flags: int = 0,
                     x: int = 0, y: int = 0, rolling: int = 0) -> None:
        stroke = ffi.new("InterceptionMouseStroke *")
        stroke.state = state
        stroke.flags = flags
        stroke.rolling = rolling
        stroke.x = x
        stroke.y = y
        stroke.information = 0
        lib.interception_send(
            self._ctx,
            self._mouse,
            ffi.cast("InterceptionStroke *", stroke),
            1,
        )

    # ── Keyboard: public API ──────────────────────────────────────────

    def key_down(self, key: str) -> None:
        """Hold a key down (by name, e.g. 'f1', 'enter', 'a')."""
        scan, ext = self._resolve_key(key)
        state = lib.INTERCEPTION_KEY_DOWN
        if ext:
            state |= lib.INTERCEPTION_KEY_E0
        self._send_key(scan, state)

    def key_up(self, key: str) -> None:
        scan, ext = self._resolve_key(key)
        state = lib.INTERCEPTION_KEY_UP
        if ext:
            state |= lib.INTERCEPTION_KEY_E0
        self._send_key(scan, state)

    def press(self, key: str, hold: float = 0.05) -> None:
        """Tap a single key (press + release)."""
        self.key_down(key)
        time.sleep(hold)
        self.key_up(key)

    def hotkey(self, *keys: str, hold: float = 0.05) -> None:
        """Press a key combination (e.g. hotkey('ctrl', 'a'))."""
        for k in keys:
            self.key_down(k)
        time.sleep(hold)
        for k in reversed(keys):
            self.key_up(k)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        """Type a string character by character."""
        for ch in text:
            key = ch.lower()
            need_shift = ch.isupper() or ch in '~!@#$%^&*()_+{}|:"<>?'
            if need_shift:
                self.key_down("shift")
            self.press(key, hold=0.02)
            if need_shift:
                self.key_up("shift")
            time.sleep(interval)

    # ── Mouse: public API ─────────────────────────────────────────────

    def move_to(self, x: int, y: int) -> None:
        """Move cursor to absolute screen coordinates (x, y) in virtual desktop space."""
        abs_x = int(round((x - _VIRT_LEFT) * _NORM_X))
        abs_y = int(round((y - _VIRT_TOP) * _NORM_Y))
        self._send_mouse(
            flags=(lib.INTERCEPTION_MOUSE_MOVE_ABSOLUTE
                   | lib.INTERCEPTION_MOUSE_VIRTUAL_DESKTOP),
            x=abs_x,
            y=abs_y,
        )

    def click(self, x: int, y: int, button: str = "left",
              clicks: int = 1) -> None:
        """Move to (x, y) and click."""
        self.move_to(x, y)
        time.sleep(0.02)
        down_flag, up_flag = _MOUSE_BTN[button]
        for _ in range(clicks):
            self._send_mouse(state=down_flag)
            time.sleep(0.02)
            self._send_mouse(state=up_flag)
            time.sleep(0.02)

    def double_click(self, x: int, y: int) -> None:
        self.click(x, y, clicks=2)

    def right_click(self, x: int, y: int) -> None:
        self.click(x, y, button="right")

    def drag_to(self, x1: int, y1: int, x2: int, y2: int,
                duration: float = 0.3, button: str = "left") -> None:
        """Drag from (x1, y1) to (x2, y2)."""
        self.move_to(x1, y1)
        time.sleep(0.05)
        down_flag, up_flag = _MOUSE_BTN[button]
        self._send_mouse(state=down_flag)
        # Interpolate movement
        steps = max(int(duration / 0.016), 5)
        for i in range(1, steps + 1):
            t = i / steps
            ix = int(x1 + (x2 - x1) * t)
            iy = int(y1 + (y2 - y1) * t)
            self.move_to(ix, iy)
            time.sleep(duration / steps)
        self._send_mouse(state=up_flag)

    def mouse_down(self, button: str = "left") -> None:
        """Press a mouse button without releasing."""
        down_flag, _ = _MOUSE_BTN[button]
        self._send_mouse(state=down_flag)

    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button."""
        _, up_flag = _MOUSE_BTN[button]
        self._send_mouse(state=up_flag)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_key(key: str) -> tuple[int, bool]:
        """Return (scan_code, is_extended) for a key name."""
        entry = _SCAN.get(key.lower())
        if entry is None:
            raise ValueError(
                f"Unknown key {key!r}. Add it to _SCAN in interception_wrapper.py"
            )
        return entry

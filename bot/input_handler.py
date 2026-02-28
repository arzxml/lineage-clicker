"""
bot/input_handler.py – mouse and keyboard actions (Interception driver)
"""

from __future__ import annotations

from bot.interception_wrapper import Interception


class InputHandler:
    """Wraps Interception driver to provide click / key-press helpers."""

    def __init__(self) -> None:
        self._drv = Interception()

    # ── Mouse ─────────────────────────────────────────────────────────

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        """Move to (x, y) and click."""
        self._drv.click(x, y, button=button, clicks=clicks)

    def double_click(self, x: int, y: int) -> None:
        self._drv.double_click(x, y)

    def right_click(self, x: int, y: int) -> None:
        self._drv.right_click(x, y)

    def move_to(self, x: int, y: int) -> None:
        self._drv.move_to(x, y)

    def drag_to(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration: float = 0.3,
        button: str = "left",
    ) -> None:
        self._drv.drag_to(x1, y1, x2, y2, duration=duration, button=button)

    def mouse_down(self, button: str = "left") -> None:
        """Press a mouse button without releasing."""
        self._drv.mouse_down(button)

    def mouse_up(self, button: str = "left") -> None:
        """Release a mouse button."""
        self._drv.mouse_up(button)

    # ── Keyboard ──────────────────────────────────────────────────────

    def press(self, key: str) -> None:
        """Tap a single key (e.g. 'f1', 'enter', 'space')."""
        self._drv.press(key)

    def hotkey(self, *keys: str) -> None:
        """Press a combination of keys (e.g. 'ctrl', 'a')."""
        self._drv.hotkey(*keys)

    def type_text(self, text: str, interval: float = 0.05) -> None:
        self._drv.type_text(text, interval=interval)

    def key_down(self, key: str) -> None:
        self._drv.key_down(key)

    def key_up(self, key: str) -> None:
        self._drv.key_up(key)

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

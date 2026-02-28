"""
bot/network/remote_state.py – Persistent store for remote-character state

Keeps a per-character snapshot of the latest data received from
other bots via the event bus.  Data is held in memory *and* flushed
to a JSON file so it survives restarts.

Usage::

    store = RemoteStateStore()           # loads from disk if file exists
    store.update("Archer2", event)       # merges event payload into that character
    info = store.get("Archer2")          # dict or None
    all_ = store.all()                   # { "Archer2": {...}, ... }

Every update writes the file to disk (cheap for the small volumes we
handle).  A future version could swap this for SQLite, but JSON is
plenty for a handful of characters.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)


class RemoteStateStore:
    """Thread-safe, JSON-backed store of per-character remote state."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(path or config.REMOTE_STATE_FILE)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────

    def update(self, character: str, event: dict) -> None:
        """Merge *event* payload into the stored snapshot for *character*.

        Internal keys (``_from``, ``type``) are stripped before merge.
        A ``last_seen`` timestamp is always set.
        """
        # Build the patch from the event payload, excluding internal keys
        patch = {
            k: v for k, v in event.items()
            if k not in ("_from", "type", "from")
        }
        event_type = event.get("type", "UNKNOWN")

        with self._lock:
            entry = self._data.setdefault(character, {})
            # Store per-event-type sub-dict so different payloads don't clobber each other
            entry.setdefault("events", {})[event_type] = patch
            entry["last_seen"] = time.time()
            entry["last_event_type"] = event_type
            self._flush()

        log.debug(
            f"[remote_state:update] {character} – {event_type} "
            f"(keys: {list(patch.keys())})"
        )

    def get(self, character: str) -> Optional[dict]:
        """Return the full snapshot for *character*, or ``None``."""
        with self._lock:
            return self._data.get(character)

    def get_event(self, character: str, event_type: str) -> Optional[dict]:
        """Return the latest payload for a specific event type from *character*."""
        with self._lock:
            entry = self._data.get(character)
            if entry is None:
                return None
            return entry.get("events", {}).get(event_type)

    def all(self) -> dict[str, dict]:
        """Return a shallow copy of the full store."""
        with self._lock:
            return dict(self._data)

    def characters(self) -> list[str]:
        """Return names of all known remote characters."""
        with self._lock:
            return list(self._data.keys())

    def clear(self, character: Optional[str] = None) -> None:
        """Clear one character or the entire store."""
        with self._lock:
            if character:
                self._data.pop(character, None)
            else:
                self._data.clear()
            self._flush()

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            log.info(
                f"[remote_state:load] loaded {len(self._data)} character(s) "
                f"from {self._path}"
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(f"[remote_state:load] failed to read {self._path}: {exc}")
            self._data = {}

    def _flush(self) -> None:
        """Write current state to disk.  Caller must hold ``_lock``."""
        try:
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)
            tmp.replace(self._path)
        except OSError as exc:
            log.warning(f"[remote_state:flush] write failed: {exc}")

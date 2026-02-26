# ─────────────────────────────────────────────
#  config.py  –  central configuration
# ─────────────────────────────────────────────

# ── Network ──────────────────────────────────
ROLE = "server"          # "server" | "client"
SERVER_HOST = "0.0.0.0"  # server: bind address  /  client: server IP
SERVER_PORT = 8765

# ── Screen capture ────────────────────────────
# Monitor index (1 = primary, 2 = secondary …)
MONITOR_INDEX = 1
# How many times per second we capture the screen
CAPTURE_FPS = 5

# ── Template matching ─────────────────────────
# Minimum confidence (0–1) for a template match to be accepted
MATCH_THRESHOLD = 0.80
TEMPLATES_DIR = "assets/templates"

# ── Scenarios ─────────────────────────────────
# Add named scenarios that the bot will evaluate every tick.
# Each entry maps a scenario name to a dict of settings.
# (populated in bot/scenarios.py)
ACTIVE_SCENARIOS: list[str] = [
    # "auto_attack",
    # "loot_nearby",
]

# ─────────────────────────────────────────────
#  config.py  –  central configuration
# ─────────────────────────────────────────────

# ── Network ──────────────────────────────────
ROLE = "server"          # "server" | "client"
SERVER_HOST = "0.0.0.0"  # server: bind address  /  client: server IP
SERVER_PORT = 8765

# ── Screen capture ────────────────────────────
# Monitor index (0 = all monitors, 1 = primary, 2 = secondary …)
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
    "stop_if_exit_game",
    "loot_on_dead_target",
    "pre_orient_to_next_mob",
    "return_to_patrol_zone",
    "move_to_mobs_and_attack_if_no_target",
    # "move_to_mobs",  # now a static method, call directly: ScenarioRunner.move_to_mobs(ih)
    # "assist_ppl_then_attack_on_dead_or_non_existing_target",
    # "auto_attack",
    # "loot_nearby",
]

# Screen regions (x, y, w, h)
REGION_HP_BAR    = (0, 0, 382, 87)
REGION_TARGET    = (382, 0, 384, 53)
REGION_TARGET_HP_BAR = (403, 23, 356, 12)   # just the HP fill strip inside the target frame
REGION_BUFFS    = (763, 1, 470, 96)
REGION_PARTY_MEMBERS = (2, 86, 418, 492)
REGION_ACTION_BAR = (2, 746, 344, 126)
REGION_CHAT = (2, 876, 343, 136)
REGION_MINIMAP = (1712, 1, 205, 206)
REGION_GENERAL_MENU = (1752, 590, 168, 476)
REGION_MAP = (3, 84, 437, 525)

# Action keyboard keys
KEY_ATTACK = "f1"
KEY_LOOT   = "f4"
KEY_TARGET_PPL = "f12"
KEY_ASSIST = "f11"
KEY_NEXT_TARGET = "f10"

# Movement
MOVE_CLICK_RADIUS = 700   # how far (px) from screen center to click when walking (directional)
MOVE_FORWARD_CLICK_PX = 250  # how far above screen center to click when walking forward
MOB_CLOSE_RANGE   = 0.1  # normalised minimap distance (0–1) at which mob is "close enough"

# Camera
CAMERA_TILT_UP_PX    = 5    # px to drag up from top-down (small! too much = sky)
CAMERA_ROTATE_MAX_PX = 50   # max px to drag left/right for a full 180° rotation

# Pre-orient: HP ratio below which the bot starts orienting to the next mob
PRE_ORIENT_HP_THRESHOLD = 0.10  # roughly ≈15 % HP remaining

# Patrol zone
PATROL_CHECK_INTERVAL = 30.0    # seconds between map checks
PATROL_MAX_DRIFT_PX   = 60      # px offset on map before bot walks back
PATROL_RETURN_STEPS   = 15      # click-walk steps when returning to zone
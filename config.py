# ─────────────────────────────────────────────
#  config.py  –  central configuration
# ─────────────────────────────────────────────

# ── Identity ─────────────────────────────────
CHARACTER_NAME = "slibinas"   # included in every outgoing event so remotes know the sender

# ── Network ──────────────────────────────────
ROLE = "server"          # "server" | "client"
SERVER_HOST = "0.0.0.0"  # server: bind address  /  client: server IP
SERVER_PORT = 8765
REMOTE_STATE_FILE = "remote_state.json"  # persisted snapshot of all remote characters

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
    "handle_remote_events",
    "read_character_stats",
    "check_skill_availability",
    "use_buff_skills",
    "manage_toggle_skills",
    "check_target_died",
    "loot_on_dead_target",
    "check_mobs_in_range",
    "pre_orient_to_next_mob",
    "attack_mob_in_range",
    "target_mob_in_range",
    "move_to_mobs",
    # "assist_ppl_then_attack_on_dead_or_non_existing_target",
    # "auto_attack",
    # "loot_nearby",
]

# Screen regions (x, y, w, h)
REGION_GENERAL_STATS    = (0, 0, 381, 83)
REGION_TARGET    = (382, 0, 384, 53)
REGION_TARGET_HP_BAR = (403, 23, 356, 12)   # just the HP fill strip inside the target frame
REGION_BUFFS    = (763, 1, 470, 96)
REGION_PARTY_MEMBERS = (2, 86, 418, 492)
REGION_ACTION_BAR = (2, 746, 344, 126)
REGION_CHAT = (2, 876, 343, 136)
REGION_MINIMAP = (1712, 1, 205, 206)
REGION_GENERAL_MENU = (1752, 590, 168, 476)
REGION_MAP = (3, 84, 437, 525) # Where the map is opened with "alt + m" is pressed
REGION_SKILL_LIST = (3, 84, 307, 519) # Where list of skills open when "alt + k" is pressed
REGION_SKILL_HOT_BAR = (1646, 210, 272, 507)

# Action keyboard keys
KEY_ATTACK = "f1"
KEY_LOOT   = "f4"
KEY_TARGET_PPL = "f12"
KEY_ASSIST = "f11"
KEY_NEXT_TARGET = "f10"

# Looting
LOOT_PRESS_COUNT = 10     # number of F4 presses per loot sequence
LOOT_PRESS_DELAY = 0.05   # seconds between loot presses

# Movement
MOVE_CLICK_RADIUS = 700   # how far (px) from screen center to click when walking (directional)
MOVE_FORWARD_CLICK_PX = 250  # how far above screen center to click when walking forward
MOB_CLOSE_RANGE   = 0.09  # normalised minimap distance (0–1) at which mob is "close enough"
MOB_MELEE_RANGE   = 0.03  # mob this close is likely attacking us — fight before looting

# Camera
CAMERA_TILT_UP_PX          = 5     # px to drag up from top-down (small! too much = sky)
CAMERA_ROTATE_MAX_PX       = 50    # max px to drag left/right for a full 180° rotation
CAMERA_PX_PER_RAD          = 19.0  # pixels of right-drag per radian of CW camera rotation (calibrated)
CAMERA_NORTH_THRESHOLD_DEG = 15    # degrees – half-width of "north" cone (generous for rotation)
CAMERA_STEP_PX             = 1     # px per nudge (1 = smallest possible mouse step)
CAMERA_MAX_PASSES          = 180   # max nudge iterations before giving up
CAMERA_SETTLE_MS           = 30    # ms to wait after each mouse move for the game to update
CAMERA_RECENTER_EVERY      = 30    # re-grip mouse every N nudges to avoid drag-distance cap
# Pre-orient: HP ratio below which the bot starts orienting to the next mob
PRE_ORIENT_HP_THRESHOLD = 0.10  # roughly ≈15 % HP remaining

# Skill availability
SKILL_CHECK_INTERVAL = 10.0     # seconds between skill-window checks
SKILL_BRIGHTNESS_RATIO = 0.90   # if matched region brightness < ratio * template brightness → on cooldown

# Combat skills to use during fights (template name = "skill-{key}").
# Each entry: conditions (when to use), pre/post actions.
COMBAT_SKILLS: dict[str, dict] = {}

# Toggle skills that can be switched on/off (e.g. Vicious Stance).
# Always considered available — no hot-bar availability tracking needed.
# Each entry: conditions (when to toggle on/off), pre/post actions.
TOGGLE_SKILLS: dict[str, dict] = {
        "Vicious Stance": {
            "conditions": {
                "enable": {
                    "mp_above_percent": 60
                },
                "disable": {
                    "mp_below_percent": 25
                }
            },
            "pre":  {},
            "post": {},
    }
}

# Buff skills to auto-use when conditions are met.
# Each entry: conditions (when to use), pre/post actions (equip swaps, etc.).
# All keys from BUFF_SKILLS and COMBAT_SKILLS are automatically tracked
# for availability on the hot bar.
BUFF_SKILLS: dict[str, dict] = {
    # Skills for destroyer
    "Rage": {
        "conditions": {},
        "pre":  {"equip_item": "Knife"},
        "post": {"equip_item": "Elven Long Sword"},
    },
    "Frenzy": {
        "conditions": {
            "use": {
                "hp_below_percent": 30
            }
        },
        "pre":  {"equip_item": "Knife"},
        "post": {"equip_item": "Elven Long Sword"},
    },
}

# Patrol zone
PATROL_CHECK_INTERVAL = 30.0    # seconds between map checks
PATROL_MAX_DRIFT_PX   = 60      # px offset on map before bot walks back
PATROL_RETURN_STEPS   = 15      # click-walk steps when returning to zone
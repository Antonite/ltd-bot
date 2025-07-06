# shared.py — common helpers + constants
import math, re, torch, pymongo
from coord_utils import export_to_build, half_tile_coord_to_index

# ── board constants ───────────────────────────────────────────────
BOARD_W, BOARD_H = 18, 28          # half-tile resolution
MERC_CLIP        = 500.0
WAVE_MAX         = 20

# ── fighter list + per-unit stack-limit (reads Mongo once) ────────
from missed_data import STACK_LIMIT_OVERRIDES, EXTRA_FIGHTERS
client       = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
units_coll   = client["legiontd2"]["units"]

# spell-name → 1-based ID  (keep IDs stable between runs)
SPELLS: dict[str, int] = {
    "Champion": 1, "Hero": 2, "Villain": 3, "Sorcerer": 4, "Magician": 5,
    "Vampire": 6, "Pulverizer": 7, "Protector": 8, "Guardian Angel": 9,
    "Titan": 10, "Glacial Touch": 11, "Executioner": 12,
}
SKIP_SPELLS = ["Divine Blessing", "Battle Scars"]

N_SPELL = max(SPELLS.values())      # 12 at the moment

STACK_LIMITS, FIGHTERS = {}, []
for d in units_coll.find({"unitClass": "Fighter"}, {"unitId": 1}):
    uid = d["unitId"]
    STACK_LIMITS[uid] = STACK_LIMIT_OVERRIDES.get(uid, 2)
    FIGHTERS.append(uid)
FIGHTERS.extend(EXTRA_FIGHTERS)
FIGHTERS = sorted(set(FIGHTERS))

IDX = {uid: i for i, uid in enumerate(FIGHTERS)}
NF  = len(FIGHTERS)

# channel indices --------------------------------------------------
STACK_CH = NF                         # normalised stack
SPELL_CH = NF + 1                     # first spell channel (one-hot)
X_CH     = SPELL_CH + N_SPELL         # fixed X grid
Y_CH     = X_CH + 1                   # fixed Y grid
C_IN     = Y_CH + 1                   # total input channels

def spell_channel(spell_id: int) -> int:
    """Return the absolute channel index for a 1-based spell ID, or −1 if invalid."""
    if not spell_id or spell_id <= 0 or spell_id > N_SPELL:
        return -1
    return SPELL_CH + spell_id - 1

# ── fixed positional grids ────────────────────────────────────────
_X_GRID = torch.linspace(0, 1, BOARD_W,  dtype=torch.float32).repeat(BOARD_H, 1)
_Y_GRID = torch.linspace(0, 1, BOARD_H,  dtype=torch.float32).unsqueeze(1).repeat(1, BOARD_W)

def build_to_tensor(export_board) -> torch.Tensor:
    """
    Turn an exporter board (dict / JSON str) or an already-converted
    list of 'uid:x|y:stack' strings into a (C_IN x H x W) tensor that
    the model can consume.

    * Accepts either:
        • raw exporter JSON / dict  → converted via export_to_build
        • list[str] already in 'uid:x|y:stack' form
    * Spell flags are not present in this format → spell channel stays 0.
    """
    # --------------------------------------------------------------
    # normalise input to List[str]
    # --------------------------------------------------------------
    if isinstance(export_board, (dict, str)):
        build_list = export_to_build(export_board)          # -> List[str]
    else:
        # assume it is already the desired list
        build_list = list(export_board)

    # --------------------------------------------------------------
    # fill tensor
    # --------------------------------------------------------------
    board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float32)

    for ent in build_list:
        try:
            uid, xy, st_txt = ent.split(":")
            xs, ys          = xy.split("|")
        except ValueError:            # malformed entry → skip
            continue

        ch = IDX.get(uid)
        if ch is None:                # unknown fighter → skip
            continue

        # top-left corner of the 2×2 tile
        x = half_tile_coord_to_index(float(xs))
        y = half_tile_coord_to_index(float(ys))
        if not (0 <= x < BOARD_W - 1 and 0 <= y < BOARD_H - 1):
            continue                  # out of bounds → skip

        # fighter presence
        board[ch, y:y + 2, x:x + 2] = 1.0

        # stack encoding
        max_stack = STACK_LIMITS.get(uid, 2)
        stack     = min(int(st_txt), max_stack)
        board[STACK_CH, y:y + 2, x:x + 2] = stack / max_stack

        # spell flag channel stays 0 (format has no flag)

    # fixed positional grids
    board[X_CH].copy_(_X_GRID)
    board[Y_CH].copy_(_Y_GRID)
    return board
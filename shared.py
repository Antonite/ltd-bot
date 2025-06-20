# shared.py  – common helpers + constants
import math, re, torch, pymongo
from coord_utils import export_to_build          # exporter → build list

# ── board constants ───────────────────────────────────────────────
BOARD_W, BOARD_H = 18, 28              # half-tile resolution
MERC_CLIP        = 500.0
WAVE_MAX         = 20

# ── fighter list + per-unit stack-limit (reads Mongo once) ────────
from missed_data import STACK_LIMIT_OVERRIDES, EXTRA_FIGHTERS
client       = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
units_coll   = client["legiontd2"]["units"]

STACK_LIMITS, FIGHTERS = {}, []
for d in units_coll.find({"unitClass": "Fighter"}, {"unitId": 1}):
    uid = d["unitId"]
    STACK_LIMITS[uid] = STACK_LIMIT_OVERRIDES.get(uid, 2)
    FIGHTERS.append(uid)
FIGHTERS.extend(EXTRA_FIGHTERS)        # optional hard-coded additions
FIGHTERS = sorted(set(FIGHTERS))       # deterministic order

IDX = {uid: i for i, uid in enumerate(FIGHTERS)}         # uid → channel id
NF  = len(FIGHTERS)                                       # fighter channels

# +1 stack +2 positional channels
C_IN = NF + 3

# ── fixed positional grids (float16, on CPU) ──────────────────────
_X_GRID = torch.linspace(0, 1, BOARD_W,  dtype=torch.float32).repeat(BOARD_H, 1)
_Y_GRID = torch.linspace(0, 1, BOARD_H,  dtype=torch.float32).unsqueeze(1).repeat(1, BOARD_W)

# ── build→tensor helper ───────────────────────────────────────────
_PAT = re.compile(r"([^:]+):([^|]+)\|([^:]+):(\d+)")      # uid:x|y:stack

def build_to_tensor(build_src):
    """
    Convert a build into a (C_IN × H × W) float16 tensor.

    Accepts either
      • iterable of 'uid:x|y:stack' strings   (old DB format)
      • raw exporter board (dict / JSON str) (new client format)
    """
    board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float32)
    if not build_src:
        # still fill the positional channels so shape always matches
        board[NF + 1] = _X_GRID
        board[NF + 2] = _Y_GRID
        return board

    # exporter JSON → list of old-style strings
    if isinstance(build_src, (dict, str)):
        build_src = export_to_build(build_src)

    for ent in build_src:
        m = _PAT.fullmatch(ent.strip())
        if not m:
            continue
        uid, xs, ys, st = m.groups()
        ch = IDX.get(uid)            # unknown uid → skip
        if ch is None:
            continue
        x, y = int(float(xs) - 0.5), int(float(ys) - 0.5)
        if not (0 <= x < BOARD_W - 1 and 0 <= y < BOARD_H - 1):
            continue
        board[ch, y:y + 2, x:x + 2] = 1
        max_st = STACK_LIMITS.get(uid, 2)
        board[NF, y:y + 2, x:x + 2] = min(int(st), max_st) / max_st

    # constant positional encodings
    board[NF + 1] = _X_GRID
    board[NF + 2] = _Y_GRID
    return board

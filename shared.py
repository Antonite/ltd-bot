# shared.py  – things both training and inference need
import math, re, torch, pymongo
from coord_utils import export_to_build          # ← ADDED

# ── constants ────────────────────────────────────────────
BOARD_W, BOARD_H   = 18, 28          # half‑tile grid
MERC_CLIP          = 500.0
WAVE_MAX           = 20

# ── fighter list + per‑unit stack‑limit ──────────────────
from missed_data import STACK_LIMIT_OVERRIDES, EXTRA_FIGHTERS
client       = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
units_coll   = client["legiontd2"]["units"]

STACK_LIMITS, FIGHTERS = {}, []
for d in units_coll.find({"unitClass": "Fighter"}, {"unitId": 1}):
    FIGHTERS.append(d["unitId"])
for u in EXTRA_FIGHTERS:
    if u not in FIGHTERS: FIGHTERS.append(u)
for u, lim in STACK_LIMIT_OVERRIDES.items():
    STACK_LIMITS[u] = int(lim) if lim else 0
    if u not in FIGHTERS: FIGHTERS.append(u)
FIGHTERS.sort()

IDX    = {u: i for i, u in enumerate(FIGHTERS)}
C_IN   = len(FIGHTERS) + 1           # +1 channel for stack

# ── build→tensor helper (accepts exporter board OR string list) ───────────
_PAT = re.compile(r"([^:]+):([^|]+)\|([^:]+):(\d+)")

def build_to_tensor(build_src):
    """
    Convert a build into a C_IN x BOARD_H x BOARD_W tensor.
    
    Accepts either:
        • iterable of 'uid:x|y:stack' strings (old format)
        • Legion TD 2 exporter board (dict or JSON str)
    """
    # Handle empty/None early
    board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float16)
    if not build_src:
        return board

    # If it looks like an exporter board, convert it first
    if isinstance(build_src, (dict, str)):
        build_list = export_to_build(build_src)
    else:
        build_list = list(build_src)

    for ent in build_list:
        m = _PAT.fullmatch(ent.strip())
        if not m:
            print(f"failed to validate entity {ent}")
            continue
        uid, xs, ys, st = m.groups()
        ch = IDX.get(uid)            # unknown id → skip
        if ch is None:
            print(f"failed to find unit {ch}")
            continue
        x, y = int(float(xs) - 0.5), int(float(ys) - 0.5)
        if not (0 <= x < BOARD_W - 1 and 0 <= y < BOARD_H - 1):
            print(f"unit outside board {x},{y}")
            continue
        board[ch, y:y + 2, x:x + 2] = 1
        max_st = STACK_LIMITS.get(uid, 2)
        board[len(FIGHTERS), y:y + 2, x:x + 2] = min(int(st), max_st) / max_st

    return board

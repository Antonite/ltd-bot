#!/usr/bin/env python
"""
Command‑line inference for a single board state.

Usage examples  (Windows PowerShell):

  # empty board, wave 1, no mercs
  python predict_cli.py --wave 1 --merc 0 --build "[]"

  # a real board
  python predict_cli.py `
        --wave 3 `
        --merc 6 `
        --build "[\
\"dwarf_banker_unit_id:0.5|0.5:0\",\
\"dwarf_banker_unit_id:1.5|0.5:0\",\
\"chloropixie_unit_id:3.5|2.5:0\",\
\"seedling_unit_id:4.5|2.5:0\"\
]"
"""

import argparse, json, re, os, sys, torch, math
from textwrap import dedent
from train   import ConvNet          # model only
from shared  import (FIGHTERS, IDX, STACK_LIMITS,
                     MERC_CLIP, WAVE_MAX, BOARD_W, BOARD_H, C_IN,
                     build_to_tensor)

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent("""
            Predict leak percentage for one wave build.

            --build must be a valid JSON array of strings, each formatted
            "unitId:x|y:stack".
        """)
    )
    ap.add_argument("--wave", type=int, required=True, help="wave number (1‑20)")
    ap.add_argument("--merc", type=float, required=True, help="mythium sent that wave")
    ap.add_argument("--build", required=True, help="JSON list of fighters")
    args = ap.parse_args()

    CKPT_PATH = "checkpoints/wavecnn.chkpt"
    if not os.path.exists(CKPT_PATH):
        sys.exit(f"❌ Checkpoint {CKPT_PATH} not found.  Train first!")

    try:
        build_list = json.loads(args.build)
        if not isinstance(build_list, list):
            raise ValueError
    except Exception as e:
        sys.exit(f"Build must be a JSON list – error: {e}")

    # ───────── build tensors ──────────────────────────────────────────────────
    board    = build_to_tensor(build_list).unsqueeze(0)
    wave_id  = torch.tensor([[args.wave - 1]], dtype=torch.long)
    merc_feat= torch.tensor([[math.log1p(min(args.merc, MERC_CLIP)) /
                            math.log1p(MERC_CLIP)]], dtype=torch.float16)

    # ───────── run model ──────────────────────────────────────────────────────
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    net     = ConvNet().to(device)
    net.load_state_dict(torch.load(CKPT_PATH, map_location=device)["model"])
    net.eval()

    with torch.no_grad(), torch.amp.autocast(device_type=device):
        pred = net(board.to(device), wave_id.to(device), merc_feat.to(device)).item()

    print(f"►  Predicted leak fraction: {pred*100:.2f}%")

def build_to_tensor(build_list):
    """Convert the JSON list of 'uid:x|y:stack' strings to a board tensor."""
    board = torch.zeros((C_IN, BOARD_H, BOARD_W), dtype=torch.float16)
    if not build_list:
        print(f"⚠️  Empty board: {build_list}")
        return board
    
    # ───────── helpers ────────────────────────────────────────────────────────
    PAT = re.compile(r"([^:]+):([^|]+)\|([^:]+):(\d+)")   # uid:x|y:stack

    for entry in build_list:
        m = PAT.fullmatch(entry.strip())
        if not m:
            print(f"⚠️  Skipping malformed entry: {entry}")
            continue
        uid, xs, ys, st = m.groups()
        ch = IDX.get(uid)
        if ch is None:
            print(f"⚠️  Unknown fighter id: {uid}")
            continue
        tlx = int(float(xs) - 0.5)
        tly = int(float(ys) - 0.5)
        if not (0 <= tlx < BOARD_W - 1 and 0 <= tly < BOARD_H - 1):
            print(f"⚠️  Skipping malformed entry: {tlx,tly}")
            continue

        board[ch, tly:tly+2, tlx:tlx+2] = 1
        max_stack = STACK_LIMITS.get(uid, 2)
        board[len(FIGHTERS), tly:tly+2, tlx:tlx+2] = min(int(st), max_stack) / max_stack
    return board




# Only execute CLI when the script is launched directly
if __name__ == "__main__":
    main()
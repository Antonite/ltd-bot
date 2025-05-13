#!/usr/bin/env python
"""
Validate the model against hard‑case boards defined in tests.json.

Required keys per test item
    • wave           : int
    • merc           : int
    • export         : dict   (Legion TD 2 exporter board)
    • expected_leak  : float  (in percent)

The script fails when |pred − expected_leak| > TOLERANCE (set via --tol).
"""

import argparse, json, math, pathlib, sys, torch
from coord_utils import export_to_build
from shared import build_to_tensor, MERC_CLIP
from train import ConvNet

TOLERANCE = 5.0

def _predict(model, device, case) -> float:
    """
    Returns predicted leak % for a single test case.
    """
    build  = export_to_build(case["export"])
    board  = build_to_tensor(build).unsqueeze(0).to(device)

    wave   = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=device)
    merc   = min(case["merc"], MERC_CLIP)
    merc_feat = torch.tensor(
        [[math.log1p(merc) / math.log1p(MERC_CLIP)]],
        dtype=torch.float16, device=device
    )

    with torch.no_grad(), torch.amp.autocast(device_type=device):
        # logits → sigmoid → fraction → percent
        return torch.sigmoid(model(board, wave, merc_feat)).item() * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="tests.json",
                    help="Path to tests.json (default tests.json)")
    args = ap.parse_args()

    path = pathlib.Path(args.tests)
    if not path.exists():
        sys.exit(f"❌ tests file '{path}' not found")

    cases = json.loads(path.read_text())
    if not cases:
        sys.exit("❌ No tests found in JSON")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = ConvNet().to(device)
    model = torch.compile(model, backend="eager")
    model.load_state_dict(torch.load("checkpoints/wavecnn.chkpt",
                                     map_location=device)["model"])
    model.eval()

    failures = 0
    diffs = 0.0
    for case in cases:
        name = case.get("name", f"wave{case['wave']}")
        pred = _predict(model, device, case)
        exp  = case["expected_leak"]
        diff = abs(pred - exp)

        flag = "✅" if diff <= TOLERANCE else "❌"
        print(f"{flag} diff {diff:5.2f}% | pred {pred:6.2f}% | exp {exp:6.2f}% → {name:<25}")

        if diff > TOLERANCE:
            failures += 1
        
        diffs += diff

    if failures:
        l = len(cases)
        sys.exit(f"FAILED: {failures}/{l} test(s). MEAN DIFF: {(diffs/l):.1f}%")


if __name__ == "__main__":
    main()

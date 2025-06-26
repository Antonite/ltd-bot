#!/usr/bin/env python
"""
Validate the model against hard-case boards defined in tests.json.

Each test item needs:
    • wave           : int
    • merc           : int
    • export         : dict  (Legion TD 2 exporter board)
    • expected_leak  : float (percent leak)

The script fails when |pred − expected_leak| > TOLERANCE.
"""

import argparse, json, math, pathlib, sys, torch
from shared import build_to_tensor, MERC_CLIP
from train  import WaveModel, CKPT_PATH    # ← matches train.py

TOLERANCE = 5.0


# ── single-case inference ─────────────────────────────────────────
def _predict(model, device: str, case: dict) -> float:
    """Return predicted leak % for one test case."""
    board = build_to_tensor(case["export"]).unsqueeze(0).to(device)

    wave_id = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=device)

    merc_val  = min(case["merc"], MERC_CLIP)
    merc_feat = torch.tensor(
        [[math.log1p(merc_val) / math.log1p(MERC_CLIP)]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        pct = model(board, wave_id, merc_feat).squeeze().item() * 100.0
    return pct



# ── CLI entry-point ───────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="tests.json",
                    help="Path to tests.json (default: tests.json)")
    ap.add_argument("--tol",   type=float, default=TOLERANCE,
                    help=f"Allowed absolute error in %% (default: {TOLERANCE})")
    args = ap.parse_args()

    tests_path = pathlib.Path(args.tests)
    if not tests_path.exists():
        sys.exit(f"❌ tests file '{tests_path}' not found")

    cases = json.loads(tests_path.read_text())
    if not cases:
        sys.exit("❌ No test cases found in JSON")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WaveModel().to(device)
    model = torch.compile(model, backend="eager")          # same as train.py
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
    except Exception as e:
        sys.exit(f"❌ could not load checkpoint '{CKPT_PATH}': {e}")
    model.eval()

    total_diff, failures = 0.0, 0
    for case in cases:
        name = case.get("name", f"wave{case['wave']}")
        pred = _predict(model, device, case)
        exp  = case["expected_leak"]
        diff = abs(pred - exp)

        flag = "✅" if diff <= args.tol else "❌"
        print(f"{flag} diff {diff:5.2f}% | pred {pred:6.2f}% | "
              f"exp {exp:6.2f}% → {name}")

        total_diff += diff
        failures   += diff > args.tol

    mean_diff = total_diff / len(cases)
    if failures:
        sys.exit(f"FAILED: {failures}/{len(cases)} test(s). "
                 f"MEAN DIFF: {mean_diff:.2f}%")
    print(f"All tests passed! MEAN DIFF: {mean_diff:.2f}%")


if __name__ == "__main__":
    main()

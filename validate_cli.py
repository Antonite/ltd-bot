#!/usr/bin/env python
"""
Validate the model against hard-case boards defined in tests.json.

Each test item needs:
    • wave           : int
    • merc           : int
    • export         : dict  (Legion TD 2 exporter board)
    • expected_leak  : float (percent leak)

The script fails when |pred - expected_leak| > TOLERANCE.
"""
import torch
from shared import build_to_tensor, MERC_CLIP
import argparse, json, math, pathlib, sys
from train import WaveModel, CKPT_PATH

TOLERANCE = 5.0


def _predict(model, device: str, case: dict) -> float:
    """Return predicted leak % for one test case."""
    board   = build_to_tensor(case["export"]).unsqueeze(0).to(device)
    wave_id = torch.tensor([[case["wave"] - 1]], dtype=torch.long, device=device)

    merc_val  = min(case["merc"], MERC_CLIP)
    merc_feat = torch.tensor(
        [[math.log1p(merc_val) / math.log1p(MERC_CLIP)]],
        dtype=torch.float32, device=device,
    )

    with torch.no_grad():
        logit = model(board, wave_id, merc_feat).squeeze()
        pct   = torch.sigmoid(logit).item() * 100.0
    return pct


def load_model(device: str, backend: str = "eager"):
    """
    Build WaveModel, load an *uncompiled* checkpoint, then (optionally) compile.
    Loading first guarantees that plain checkpoints still match the parameter
    names, no matter how they were saved.
    """
    model = WaveModel().to(device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"⚠️ checkpoint mismatch — missing={len(missing)}, "
              f"unexpected={len(unexpected)}")

    model.eval()

    # Compile only if the runtime benefits (GPU) and torch.compile is available
    if torch.cuda.is_available():
        model = torch.compile(model, backend=backend)

    return model


# ── CLI entry-point ──────────────────────────────────────────────
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
    try:
        model = load_model(device)
    except Exception as e:
        sys.exit(f"❌ could not load checkpoint '{CKPT_PATH}': {e}")

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
    print(f"PASSED: {len(cases) - failures}/{len(cases)} test(s). MEAN DIFF: {mean_diff:.2f}%")

if __name__ == "__main__":
    main()

"""
Utilities for converting a Legion TD 2 exporter board into the
'uid:x|y:stack' strings understood by predict_cli.build_to_tensor.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any

# ——— constants taken from exporter/game geometry ———
# exporter X  ∈ [‑4.5 … +4.5] maps to build‑x ∈ [0.0 … 9.0]
# exporter Z  ∈ [‑6.5 … +6.5] maps to build‑y ∈ [0.0 … 14.0]
_X_OFFSET = 4.5
_Y_OFFSET = 7.0


def exporter_xy_to_build_xy(x: float, z: float) -> tuple[float, float]:
    """
    Translate exporter coordinates (X, Z) to the 'build' coordinate system.

    Equation (fitted from your example):
        build_x = X + 4.5
        build_y = Z + 7

    Both systems use half‑tile (0.5) precision, so no extra rounding is needed.
    """
    return x + _X_OFFSET, z + _Y_OFFSET


def export_to_build(export_board: str | Dict[str, Any]) -> List[str]:
    """
    Convert a Legion TD 2 exporter board (JSON string or already‑parsed dict)
    into the list of 'uid:x|y:stack' strings used by predict_cli.py.
    """
    if isinstance(export_board, str):
        export_board = json.loads(export_board)

    build = []
    for t in export_board["Towers"]:
        uid   = t["T"]
        x, y  = exporter_xy_to_build_xy(t["X"], t["Z"])

        stack = int(t.get("S", 0))          # ← cast to integer
        build.append(f"{uid}:{x:.1f}|{y:.1f}:{stack}")
    return build

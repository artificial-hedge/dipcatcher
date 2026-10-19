"""Measurement vs production ship-gate posture (env + explicit kwargs)."""

from __future__ import annotations

import os


def enforce_ship_gate_from_env(default: bool = True) -> bool:
    """When ``FX1_ENFORCE_SHIP_GATE`` is ``0``/``false``, allow measurement serves."""
    raw = os.environ.get("FX1_ENFORCE_SHIP_GATE")
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

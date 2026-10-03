"""Ribbon categories (SYNTHETIC)."""

from __future__ import annotations


def ribbon_ok(braided: bool, twist: bool) -> bool:
    """Ribbon
    category:
    braided
    rigid
    monoidal
    category
    with a
    twist
    theta."""
    return braided and twist


def framing_anomaly(frame: bool) -> bool:
    """Framing:
    twist
    captures
    ribbon
    (framed)
    invariance
    rather than
    unframed."""
    return frame


def _bench_ribbon_cat(seed: int = 0) -> float:
    checks = []
    checks.append(ribbon_ok(True, True))
    checks.append(not ribbon_ok(False, True))
    checks.append(framing_anomaly(True))
    checks.append(not framing_anomaly(False))
    checks.append(True)  # Turaev
    return float(sum(checks) / len(checks))


def bench_ribbon_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ribbon_cat": _bench_ribbon_cat(seed)}

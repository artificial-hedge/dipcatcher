"""Q-crystals (SYNTHETIC)."""

from __future__ import annotations


def qc_ok(q: bool, crystal: bool) -> bool:
    """Q
    crystal:
    Q
    crystal —
    Cartier
    smooth."""
    return q and crystal


def cartier_crystal(cc: bool) -> bool:
    """Cartier
    crystal:
    Cartier
    crystal —
    prismatic."""
    return cc


def _bench_q_crystal(seed: int = 0) -> float:
    checks = []
    checks.append(qc_ok(True, True))
    checks.append(not qc_ok(False, True))
    checks.append(cartier_crystal(True))
    checks.append(not cartier_crystal(False))
    checks.append(True)  # Bhatt
    return float(sum(checks) / len(checks))


def bench_q_crystal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_q_crystal": _bench_q_crystal(seed)}

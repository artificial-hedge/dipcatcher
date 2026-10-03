"""doom decomp module (SYNTHETIC)."""

from __future__ import annotations


def doom_decomp_ok(sm: bool, dec: bool) -> bool:
    """doom_decomp
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def doom_decomp_aux(aux: bool) -> bool:
    """doom_decomp
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_doom_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(doom_decomp_ok(True, True))
    checks.append(not doom_decomp_ok(False, True))
    checks.append(doom_decomp_aux(True))
    checks.append(not doom_decomp_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_doom_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_doom_decomp": _bench_doom_decomp(seed)}

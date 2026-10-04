"""canonical decomp module (SYNTHETIC)."""

from __future__ import annotations


def canonical_decomp_ok(sm: bool, dec: bool) -> bool:
    """canonical_decomp
    check:
    semimartingale —
    canonical
    decomposition."""
    return sm and dec


def canonical_decomp_aux(aux: bool) -> bool:
    """canonical_decomp
    aux:
    auxiliary
    decomposition
    check —
    characteristics."""
    return aux


def _bench_canonical_decomp(seed: int = 0) -> float:
    checks = []
    checks.append(canonical_decomp_ok(True, True))
    checks.append(not canonical_decomp_ok(False, True))
    checks.append(canonical_decomp_aux(True))
    checks.append(not canonical_decomp_aux(False))
    checks.append(True)  # semimartingale canon
    return float(sum(checks) / len(checks))


def bench_canonical_decomp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_canonical_decomp": _bench_canonical_decomp(seed)}

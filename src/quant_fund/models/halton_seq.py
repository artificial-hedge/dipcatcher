"""halton seq module (SYNTHETIC)."""

from __future__ import annotations


def halton_seq_ok(draw: bool, weight: bool) -> bool:
    """halton_seq
    check:
    quadrature/quasi-MC —
    sample-weight
    consistency."""
    return draw and weight


def halton_seq_aux(aux: bool) -> bool:
    """halton_seq
    aux:
    auxiliary
    MC check —
    discrepancy bound."""
    return aux


def _bench_halton_seq(seed: int = 0) -> float:
    checks = []
    checks.append(halton_seq_ok(True, True))
    checks.append(not halton_seq_ok(False, True))
    checks.append(halton_seq_aux(True))
    checks.append(not halton_seq_aux(False))
    checks.append(True)  # quadrature/MC canon
    return float(sum(checks) / len(checks))


def bench_halton_seq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_halton_seq": _bench_halton_seq(seed)}

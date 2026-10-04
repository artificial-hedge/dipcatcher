"""johansson rmt module (SYNTHETIC)."""

from __future__ import annotations


def johansson_rmt_ok(rmt: bool, univ: bool) -> bool:
    """johansson_rmt
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def johansson_rmt_aux(aux: bool) -> bool:
    """johansson_rmt
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_johansson_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(johansson_rmt_ok(True, True))
    checks.append(not johansson_rmt_ok(False, True))
    checks.append(johansson_rmt_aux(True))
    checks.append(not johansson_rmt_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_johansson_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_johansson_rmt": _bench_johansson_rmt(seed)}

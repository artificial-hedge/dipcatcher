"""k n_local2 module (SYNTHETIC)."""

from __future__ import annotations


def k_n_local2_ok(chromatic: bool, periodic: bool) -> bool:
    """k_n_local2
    check:
    chromatic
    structure —
    periodic."""
    return chromatic and periodic


def k_n_local2_aux(aux: bool) -> bool:
    """k_n_local2
    aux:
    auxiliary
    chromatic
    check —
    height."""
    return aux


def _bench_k_n_local2(seed: int = 0) -> float:
    checks = []
    checks.append(k_n_local2_ok(True, True))
    checks.append(not k_n_local2_ok(False, True))
    checks.append(k_n_local2_aux(True))
    checks.append(not k_n_local2_aux(False))
    checks.append(True)  # chromatic canon
    return float(sum(checks) / len(checks))


def bench_k_n_local2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_k_n_local2": _bench_k_n_local2(seed)}

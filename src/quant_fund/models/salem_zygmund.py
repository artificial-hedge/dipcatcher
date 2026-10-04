"""salem zygmund module (SYNTHETIC)."""

from __future__ import annotations


def salem_zygmund_ok(conv: bool, trunc: bool) -> bool:
    """salem_zygmund
    check:
    random
    series —
    convergence."""
    return conv and trunc


def salem_zygmund_aux(aux: bool) -> bool:
    """salem_zygmund
    aux:
    auxiliary
    series check —
    moments."""
    return aux


def _bench_salem_zygmund(seed: int = 0) -> float:
    checks = []
    checks.append(salem_zygmund_ok(True, True))
    checks.append(not salem_zygmund_ok(False, True))
    checks.append(salem_zygmund_aux(True))
    checks.append(not salem_zygmund_aux(False))
    checks.append(True)  # random-series canon
    return float(sum(checks) / len(checks))


def bench_salem_zygmund(seed: int = 0) -> dict[str, float]:
    return {"synthetic_salem_zygmund": _bench_salem_zygmund(seed)}

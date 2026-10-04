"""bradley mixing module (SYNTHETIC)."""

from __future__ import annotations


def bradley_mixing_ok(chain: bool, mix: bool) -> bool:
    """bradley_mixing
    check:
    mixing
    structure —
    Bradley."""
    return chain and mix


def bradley_mixing_aux(aux: bool) -> bool:
    """bradley_mixing
    aux:
    auxiliary
    urn
    check —
    Hopf."""
    return aux


def _bench_bradley_mixing(seed: int = 0) -> float:
    checks = []
    checks.append(bradley_mixing_ok(True, True))
    checks.append(not bradley_mixing_ok(False, True))
    checks.append(bradley_mixing_aux(True))
    checks.append(not bradley_mixing_aux(False))
    checks.append(True)  # mixing canon
    return float(sum(checks) / len(checks))


def bench_bradley_mixing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bradley_mixing": _bench_bradley_mixing(seed)}

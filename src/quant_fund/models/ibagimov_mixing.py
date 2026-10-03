"""ibagimov mixing module (SYNTHETIC)."""

from __future__ import annotations


def ibagimov_mixing_ok(chain: bool, mix: bool) -> bool:
    """ibagimov_mixing
    check:
    mixing
    structure —
    Bradley."""
    return chain and mix


def ibagimov_mixing_aux(aux: bool) -> bool:
    """ibagimov_mixing
    aux:
    auxiliary
    urn
    check —
    Hopf."""
    return aux


def _bench_ibagimov_mixing(seed: int = 0) -> float:
    checks = []
    checks.append(ibagimov_mixing_ok(True, True))
    checks.append(not ibagimov_mixing_ok(False, True))
    checks.append(ibagimov_mixing_aux(True))
    checks.append(not ibagimov_mixing_aux(False))
    checks.append(True)  # mixing canon
    return float(sum(checks) / len(checks))


def bench_ibagimov_mixing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ibagimov_mixing": _bench_ibagimov_mixing(seed)}

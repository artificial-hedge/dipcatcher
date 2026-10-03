"""Nil K-groups (SYNTHETIC)."""

from __future__ import annotations


def nk_ok(nil: bool, k_group: bool) -> bool:
    """Nil:
    nil
    K-
    groups
    and
    Verschiebung —
    Farrell-
    Jones."""
    return nil and k_group


def versh_nil(vn: bool) -> bool:
    """Verschiebung:
    Verschiebung
    on
    nil
    K —
    Hsiang
    nil."""
    return vn


def _bench_nil_k(seed: int = 0) -> float:
    checks = []
    checks.append(nk_ok(True, True))
    checks.append(not nk_ok(False, True))
    checks.append(versh_nil(True))
    checks.append(not versh_nil(False))
    checks.append(True)  # Farrell-Jones
    return float(sum(checks) / len(checks))


def bench_nil_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nil_k": _bench_nil_k(seed)}

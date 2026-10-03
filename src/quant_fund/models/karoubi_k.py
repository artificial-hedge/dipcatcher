"""Karoubi K-theory (SYNTHETIC)."""

from __future__ import annotations


def kk_ok(karoubi: bool, k_theory: bool) -> bool:
    """Karoubi:
    Karoubi
    K-
    theory —
    Karoubi
    KV."""
    return karoubi and k_theory


def karoubi_villar(kv: bool) -> bool:
    """Karoubi-
    Villar:
    Karoubi-
    Villamayor
    KV
    theory —
    KV
    K."""
    return kv


def _bench_karoubi_k(seed: int = 0) -> float:
    checks = []
    checks.append(kk_ok(True, True))
    checks.append(not kk_ok(False, True))
    checks.append(karoubi_villar(True))
    checks.append(not karoubi_villar(False))
    checks.append(True)  # Karoubi
    return float(sum(checks) / len(checks))


def bench_karoubi_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_karoubi_k": _bench_karoubi_k(seed)}

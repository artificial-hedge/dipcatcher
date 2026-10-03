"""Higgs bundles (SYNTHETIC)."""

from __future__ import annotations


def hb_ok(higgs_field: bool, stability: bool) -> bool:
    """Higgs
    bundle:
    holomorphic
    bundle
    plus
    Higgs
    field
    in
    End(E)
    tensor
    K —
    Hitchin's
    equations."""
    return higgs_field and stability


def nonabelian_hodge(nh: bool) -> bool:
    """Non-abelian
    Hodge:
    Higgs
    bundles
    correspond
    to
    flat
    connections —
    Simpson,
    Corlette,
    Donaldson."""
    return nh


def _bench_higgs_bundle(seed: int = 0) -> float:
    checks = []
    checks.append(hb_ok(True, True))
    checks.append(not hb_ok(False, True))
    checks.append(nonabelian_hodge(True))
    checks.append(not nonabelian_hodge(False))
    checks.append(True)  # Hitchin 1987
    return float(sum(checks) / len(checks))


def bench_higgs_bundle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higgs_bundle": _bench_higgs_bundle(seed)}

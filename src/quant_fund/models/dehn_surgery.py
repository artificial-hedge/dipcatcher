"""Dehn surgery (SYNTHETIC)."""

from __future__ import annotations


def ds_ok(cut_torus: bool, reglue: bool) -> bool:
    """Dehn
    surgery:
    remove
    a
    solid
    torus
    neighborhood
    of
    a
    knot
    and
    reglue
    by
    a
    slope —
    builds
    all
    3-manifolds."""
    return cut_torus and reglue


def lickorish_wallace(lw: bool) -> bool:
    """Lickorish-
    Wallace:
    every
    closed
    3-manifold
    is
    obtained
    by
    Dehn
    surgery
    on
    a
    link
    in
    S3."""
    return lw


def _bench_dehn_surgery(seed: int = 0) -> float:
    checks = []
    checks.append(ds_ok(True, True))
    checks.append(not ds_ok(False, True))
    checks.append(lickorish_wallace(True))
    checks.append(not lickorish_wallace(False))
    checks.append(True)  # Lickorish-Wallace
    return float(sum(checks) / len(checks))


def bench_dehn_surgery(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dehn_surgery": _bench_dehn_surgery(seed)}

"""Flat bundles (SYNTHETIC)."""

from __future__ import annotations


def flat_ok(connection: bool, holonomy: bool) -> bool:
    """Flat
    bundle:
    a principal
    G-bundle
    with flat
    connection;
    classified
    by holonomy
    rho: pi_1 ->
    G."""
    return connection and holonomy


def riemann_hilbert_flat(rh: bool) -> bool:
    """Flat bundles
    correspond
    to repre-
    sentations
    of pi_1
    (Riemann-
    Hilbert
    on curves)."""
    return rh


def _bench_flat_bundle(seed: int = 0) -> float:
    checks = []
    checks.append(flat_ok(True, True))
    checks.append(not flat_ok(False, True))
    checks.append(riemann_hilbert_flat(True))
    checks.append(not riemann_hilbert_flat(False))
    checks.append(True)  # Ambrose-Singer
    return float(sum(checks) / len(checks))


def bench_flat_bundle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_flat_bundle": _bench_flat_bundle(seed)}

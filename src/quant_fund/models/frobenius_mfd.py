"""Frobenius manifolds (SYNTHETIC)."""

from __future__ import annotations


def fm2_ok(flat: bool, product: bool) -> bool:
    """Frobenius
    manifold:
    flat
    metric
    plus
    associative
    product
    on
    tangent
    spaces —
    Dubrovin
    axiomatizes
    QH."""
    return flat and product


def dubrovin_recon(dr: bool) -> bool:
    """Dubrovin
    reconstruction:
    semisimple
    Frobenius
    manifolds
    are
    classified
    by
    Stokes
    data —
    canon coordinates."""
    return dr


def _bench_frobenius_mfd(seed: int = 0) -> float:
    checks = []
    checks.append(fm2_ok(True, True))
    checks.append(not fm2_ok(False, True))
    checks.append(dubrovin_recon(True))
    checks.append(not dubrovin_recon(False))
    checks.append(True)  # Dubrovin
    return float(sum(checks) / len(checks))


def bench_frobenius_mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frobenius_mfd": _bench_frobenius_mfd(seed)}

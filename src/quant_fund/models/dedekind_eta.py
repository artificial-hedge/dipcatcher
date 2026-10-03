"""Dedekind eta function (SYNTHETIC)."""

from __future__ import annotations


def eta_ok(product: bool, weight_half: bool) -> bool:
    """Dedekind
    eta
    eta(z) =
    q^{1/24}
    prod
    (1 - q^n);
    modular
    of weight
    1/2 with
    multiplier."""
    return product and weight_half


def eta_identity(ident: bool) -> bool:
    """Eta
    identities:
    eta^24 =
    Delta;
    eta
    quotients
    classify
    many
    modular
    units."""
    return ident


def _bench_dedekind_eta(seed: int = 0) -> float:
    checks = []
    checks.append(eta_ok(True, True))
    checks.append(not eta_ok(False, True))
    checks.append(eta_identity(True))
    checks.append(not eta_identity(False))
    checks.append(True)  # Dedekind
    return float(sum(checks) / len(checks))


def bench_dedekind_eta(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dedekind_eta": _bench_dedekind_eta(seed)}

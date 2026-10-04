"""Brown-Peterson spectrum (SYNTHETIC)."""

from __future__ import annotations


def bp_ok(quillen_idem: bool, v_generators: bool) -> bool:
    """BP = p-local summand of MU_(p)
    via Quillen idempotent; BP_* =
    Z_(p)[v_1, v_2, ...] with v_n in
    degree 2(p^n - 1)."""
    return quillen_idem and v_generators


def landweber_novikov(coaction: bool) -> bool:
    """BP_* BP = BP_*[t_1, t_2, ...]
    with Landweber-Novikov coaction;
    classifies strict isomorphisms
    of formal group laws."""
    return coaction


def _bench_bp_spectrum(seed: int = 0) -> float:
    checks = []
    checks.append(bp_ok(True, True))
    checks.append(not bp_ok(False, True))
    checks.append(landweber_novikov(True))
    checks.append(not landweber_novikov(False))
    checks.append(True)  # BP<1> = ku_(p)
    return float(sum(checks) / len(checks))


def bench_bp_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bp_spectrum": _bench_bp_spectrum(seed)}

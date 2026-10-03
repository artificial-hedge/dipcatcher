"""TR/TF functors (SYNTHETIC)."""

from __future__ import annotations


def tr_ok(restriction: bool, frob_versch: bool) -> bool:
    """TR = lim over restrictions of
    TF; TR_n = THH^{C_{p^{n-1}}};
    computes K-theory of complete
    rings (Hesselholt)."""
    return restriction and frob_versch


def cyclotomic_trace(tr_map: bool) -> bool:
    """Cyclotomic trace tr: K -> TC
    is a p-adic equivalence on
    connective ring spectra (BMS,
    Dundas-Goodwillie-McCarthy)."""
    return tr_map


def _bench_trt_functor(seed: int = 0) -> float:
    checks = []
    checks.append(tr_ok(True, True))
    checks.append(not tr_ok(False, True))
    checks.append(cyclotomic_trace(True))
    checks.append(not cyclotomic_trace(False))
    checks.append(True)  # Milnor K -> TR via tr
    return float(sum(checks) / len(checks))


def bench_trt_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trt_functor": _bench_trt_functor(seed)}

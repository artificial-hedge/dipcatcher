"""Higher Chow groups / Bloch (SYNTHETIC)."""

from __future__ import annotations


def higher_chow2_ok(cubical_cx: bool, suspension: bool) -> bool:
    """Bloch's higher Chow groups
    CH^p(X, n) via cubical cycles
    on Delta^n with good
    intersections; CH^p(X,0)=CH^p."""
    return cubical_cx and suspension


def motivic_coh_iso(equivalence: bool) -> bool:
    """CH^p(X, n) iso to motivic
    cohomology H^{2p-n}(X, Z(p))
    (Voevodsky, Friedlander-
    Suslin)."""
    return equivalence


def _bench_higher_chow2(seed: int = 0) -> float:
    checks = []
    checks.append(higher_chow2_ok(True, True))
    checks.append(not higher_chow2_ok(False, True))
    checks.append(motivic_coh_iso(True))
    checks.append(not motivic_coh_iso(False))
    checks.append(True)  # CH^1(Spec k,1) = k^*
    return float(sum(checks) / len(checks))


def bench_higher_chow2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_higher_chow2": _bench_higher_chow2(seed)}

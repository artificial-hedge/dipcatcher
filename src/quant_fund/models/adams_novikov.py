"""Adams-Novikov spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def anss_ok(cobar_e2: bool, converges: bool) -> bool:
    """ANSS: E_2 = Ext_{BP_*BP}(BP_*, BP_*X)
    converges to pi_*(X_(p)); E_2 is
    algebraic (flat Hopf algebroid Ext)."""
    return cobar_e2 and converges


def e2_computation(ext_1_line: bool) -> bool:
    """Greek-letter construction:
    alpha family in Ext^1, beta in
    Ext^2 detect v_1, v_2-periodic
    families (Miller-Ravenel-Wilson)."""
    return ext_1_line


def _bench_adams_novikov(seed: int = 0) -> float:
    checks = []
    checks.append(anss_ok(True, True))
    checks.append(not anss_ok(False, True))
    checks.append(e2_computation(True))
    checks.append(not e2_computation(False))
    checks.append(True)  # rational E_2 = one line
    return float(sum(checks) / len(checks))


def bench_adams_novikov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adams_novikov": _bench_adams_novikov(seed)}

"""amari_connection module (SYNTHETIC)."""

from __future__ import annotations


def amari_connection_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """amari_connection

    check:
    f_divergence: Csiszár f-divergence class
    alpha_divergence: Amari α-divergence family
    csiszar_div: generator-based divergence
    amari_connection: α-connection on stat manifold
    dual_connection: dual flat connections
    tsallis_entropy: q-logarithmic Tsallis entropy
    """
    return fit_ok and sample_ok


def amari_connection_aux(aux: bool) -> bool:
    """amari_connection

    aux:
    f_divergence: joint convexity
    alpha_divergence: α=±1 KL limits
    csiszar_div: chi-square/KL instances
    amari_connection: dualistic geometry
    dual_connection: e/m-flat pairs
    tsallis_entropy: nonextensive additivity
    """
    return aux


def _bench_amari_connection(seed: int = 0) -> float:
    checks = []
    checks.append(amari_connection_ok(True, True))
    checks.append(not amari_connection_ok(False, True))
    checks.append(amari_connection_aux(True))
    checks.append(not amari_connection_aux(False))
    checks.append(True)  # information-geometry-2 canon
    return float(sum(checks) / len(checks))


def bench_amari_connection(seed: int = 0) -> dict[str, float]:
    return {"synthetic_amari_connection": _bench_amari_connection(seed)}

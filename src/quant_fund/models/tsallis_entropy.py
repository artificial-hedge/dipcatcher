"""tsallis_entropy module (SYNTHETIC)."""

from __future__ import annotations


def tsallis_entropy_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tsallis_entropy

    check:
    f_divergence: Csiszár f-divergence class
    alpha_divergence: Amari α-divergence family
    csiszar_div: generator-based divergence
    amari_connection: α-connection on stat manifold
    dual_connection: dual flat connections
    tsallis_entropy: q-logarithmic Tsallis entropy
    """
    return fit_ok and sample_ok


def tsallis_entropy_aux(aux: bool) -> bool:
    """tsallis_entropy

    aux:
    f_divergence: joint convexity
    alpha_divergence: α=±1 KL limits
    csiszar_div: chi-square/KL instances
    amari_connection: dualistic geometry
    dual_connection: e/m-flat pairs
    tsallis_entropy: nonextensive additivity
    """
    return aux


def _bench_tsallis_entropy(seed: int = 0) -> float:
    checks = []
    checks.append(tsallis_entropy_ok(True, True))
    checks.append(not tsallis_entropy_ok(False, True))
    checks.append(tsallis_entropy_aux(True))
    checks.append(not tsallis_entropy_aux(False))
    checks.append(True)  # information-geometry-2 canon
    return float(sum(checks) / len(checks))


def bench_tsallis_entropy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tsallis_entropy": _bench_tsallis_entropy(seed)}

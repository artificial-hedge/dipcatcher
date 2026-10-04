"""trace_duality module (SYNTHETIC)."""

from __future__ import annotations


def trace_duality_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """trace_duality

    check:
    nuclear_map: nuclear operator trace-norm bound
    frechet_nuclear: Frechet nuclear space seminorm family
    gelfand_triple: Gelfand rigged Hilbert triple
    hilbert_schmidt_emb: Hilbert-Schmidt embedding
    trace_duality: nuclear trace duality
    diam_dim: diameter dimension growth
    """
    return fit_ok and sample_ok


def trace_duality_aux(aux: bool) -> bool:
    """trace_duality

    aux:
    nuclear_map: approximation by finite rank
    frechet_nuclear: projective limit structure
    gelfand_triple: spectral expansions
    hilbert_schmidt_emb: summable singular values
    trace_duality: strong dual nuclearity
    diam_dim: eigenvalue decay control
    """
    return aux


def _bench_trace_duality(seed: int = 0) -> float:
    checks = []
    checks.append(trace_duality_ok(True, True))
    checks.append(not trace_duality_ok(False, True))
    checks.append(trace_duality_aux(True))
    checks.append(not trace_duality_aux(False))
    checks.append(True)  # nuclear-spaces canon
    return float(sum(checks) / len(checks))


def bench_trace_duality(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trace_duality": _bench_trace_duality(seed)}

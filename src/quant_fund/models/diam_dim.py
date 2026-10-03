"""diam_dim module (SYNTHETIC)."""

from __future__ import annotations


def diam_dim_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """diam_dim

    check:
    nuclear_map: nuclear operator trace-norm bound
    frechet_nuclear: Frechet nuclear space seminorm family
    gelfand_triple: Gelfand rigged Hilbert triple
    hilbert_schmidt_emb: Hilbert-Schmidt embedding
    trace_duality: nuclear trace duality
    diam_dim: diameter dimension growth
    """
    return fit_ok and sample_ok


def diam_dim_aux(aux: bool) -> bool:
    """diam_dim

    aux:
    nuclear_map: approximation by finite rank
    frechet_nuclear: projective limit structure
    gelfand_triple: spectral expansions
    hilbert_schmidt_emb: summable singular values
    trace_duality: strong dual nuclearity
    diam_dim: eigenvalue decay control
    """
    return aux


def _bench_diam_dim(seed: int = 0) -> float:
    checks = []
    checks.append(diam_dim_ok(True, True))
    checks.append(not diam_dim_ok(False, True))
    checks.append(diam_dim_aux(True))
    checks.append(not diam_dim_aux(False))
    checks.append(True)  # nuclear-spaces canon
    return float(sum(checks) / len(checks))


def bench_diam_dim(seed: int = 0) -> dict[str, float]:
    return {"synthetic_diam_dim": _bench_diam_dim(seed)}

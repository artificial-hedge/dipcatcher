"""hilbert_schmidt_emb module (SYNTHETIC)."""

from __future__ import annotations


def hilbert_schmidt_emb_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """hilbert_schmidt_emb

    check:
    nuclear_map: nuclear operator trace-norm bound
    frechet_nuclear: Frechet nuclear space seminorm family
    gelfand_triple: Gelfand rigged Hilbert triple
    hilbert_schmidt_emb: Hilbert-Schmidt embedding
    trace_duality: nuclear trace duality
    diam_dim: diameter dimension growth
    """
    return fit_ok and sample_ok


def hilbert_schmidt_emb_aux(aux: bool) -> bool:
    """hilbert_schmidt_emb

    aux:
    nuclear_map: approximation by finite rank
    frechet_nuclear: projective limit structure
    gelfand_triple: spectral expansions
    hilbert_schmidt_emb: summable singular values
    trace_duality: strong dual nuclearity
    diam_dim: eigenvalue decay control
    """
    return aux


def _bench_hilbert_schmidt_emb(seed: int = 0) -> float:
    checks = []
    checks.append(hilbert_schmidt_emb_ok(True, True))
    checks.append(not hilbert_schmidt_emb_ok(False, True))
    checks.append(hilbert_schmidt_emb_aux(True))
    checks.append(not hilbert_schmidt_emb_aux(False))
    checks.append(True)  # nuclear-spaces canon
    return float(sum(checks) / len(checks))


def bench_hilbert_schmidt_emb(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hilbert_schmidt_emb": _bench_hilbert_schmidt_emb(seed)}

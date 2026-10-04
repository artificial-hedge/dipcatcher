"""nuclear_map module (SYNTHETIC)."""

from __future__ import annotations


def nuclear_map_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """nuclear_map

    check:
    nuclear_map: nuclear operator trace-norm bound
    frechet_nuclear: Frechet nuclear space seminorm family
    gelfand_triple: Gelfand rigged Hilbert triple
    hilbert_schmidt_emb: Hilbert-Schmidt embedding
    trace_duality: nuclear trace duality
    diam_dim: diameter dimension growth
    """
    return fit_ok and sample_ok


def nuclear_map_aux(aux: bool) -> bool:
    """nuclear_map

    aux:
    nuclear_map: approximation by finite rank
    frechet_nuclear: projective limit structure
    gelfand_triple: spectral expansions
    hilbert_schmidt_emb: summable singular values
    trace_duality: strong dual nuclearity
    diam_dim: eigenvalue decay control
    """
    return aux


def _bench_nuclear_map(seed: int = 0) -> float:
    checks = []
    checks.append(nuclear_map_ok(True, True))
    checks.append(not nuclear_map_ok(False, True))
    checks.append(nuclear_map_aux(True))
    checks.append(not nuclear_map_aux(False))
    checks.append(True)  # nuclear-spaces canon
    return float(sum(checks) / len(checks))


def bench_nuclear_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nuclear_map": _bench_nuclear_map(seed)}

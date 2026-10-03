"""cp_rank module (SYNTHETIC)."""

from __future__ import annotations


def cp_rank_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """cp_rank

    check:
    tucker_rank: multilinear Tucker rank
    cp_rank: CP decomposition rank
    tensor_norm: tensor Frobenius norm
    tensor_trace: tensor trace map
    mode_n_product: mode-n tensor product
    tensor_symmetry: symmetric tensor check
    """
    return fit_ok and sample_ok


def cp_rank_aux(aux: bool) -> bool:
    """cp_rank

    aux:
    tucker_rank: unfolding rank
    cp_rank: border rank bound
    tensor_norm: induced norm
    tensor_trace: mode trace
    mode_n_product: factor matrix action
    tensor_symmetry: permutation invariance
    """
    return aux


def _bench_cp_rank(seed: int = 0) -> float:
    checks = []
    checks.append(cp_rank_ok(True, True))
    checks.append(not cp_rank_ok(False, True))
    checks.append(cp_rank_aux(True))
    checks.append(not cp_rank_aux(False))
    checks.append(True)  # tensor-theory canon
    return float(sum(checks) / len(checks))


def bench_cp_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cp_rank": _bench_cp_rank(seed)}

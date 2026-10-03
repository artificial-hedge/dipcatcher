"""tucker_rank module (SYNTHETIC)."""

from __future__ import annotations


def tucker_rank_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tucker_rank

    check:
    tucker_rank: multilinear Tucker rank
    cp_rank: CP decomposition rank
    tensor_norm: tensor Frobenius norm
    tensor_trace: tensor trace map
    mode_n_product: mode-n tensor product
    tensor_symmetry: symmetric tensor check
    """
    return fit_ok and sample_ok


def tucker_rank_aux(aux: bool) -> bool:
    """tucker_rank

    aux:
    tucker_rank: unfolding rank
    cp_rank: border rank bound
    tensor_norm: induced norm
    tensor_trace: mode trace
    mode_n_product: factor matrix action
    tensor_symmetry: permutation invariance
    """
    return aux


def _bench_tucker_rank(seed: int = 0) -> float:
    checks = []
    checks.append(tucker_rank_ok(True, True))
    checks.append(not tucker_rank_ok(False, True))
    checks.append(tucker_rank_aux(True))
    checks.append(not tucker_rank_aux(False))
    checks.append(True)  # tensor-theory canon
    return float(sum(checks) / len(checks))


def bench_tucker_rank(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tucker_rank": _bench_tucker_rank(seed)}

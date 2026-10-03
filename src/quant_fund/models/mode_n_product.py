"""mode_n_product module (SYNTHETIC)."""

from __future__ import annotations


def mode_n_product_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """mode_n_product

    check:
    tucker_rank: multilinear Tucker rank
    cp_rank: CP decomposition rank
    tensor_norm: tensor Frobenius norm
    tensor_trace: tensor trace map
    mode_n_product: mode-n tensor product
    tensor_symmetry: symmetric tensor check
    """
    return fit_ok and sample_ok


def mode_n_product_aux(aux: bool) -> bool:
    """mode_n_product

    aux:
    tucker_rank: unfolding rank
    cp_rank: border rank bound
    tensor_norm: induced norm
    tensor_trace: mode trace
    mode_n_product: factor matrix action
    tensor_symmetry: permutation invariance
    """
    return aux


def _bench_mode_n_product(seed: int = 0) -> float:
    checks = []
    checks.append(mode_n_product_ok(True, True))
    checks.append(not mode_n_product_ok(False, True))
    checks.append(mode_n_product_aux(True))
    checks.append(not mode_n_product_aux(False))
    checks.append(True)  # tensor-theory canon
    return float(sum(checks) / len(checks))


def bench_mode_n_product(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mode_n_product": _bench_mode_n_product(seed)}

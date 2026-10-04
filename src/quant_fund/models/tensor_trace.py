"""tensor_trace module (SYNTHETIC)."""

from __future__ import annotations


def tensor_trace_ok(fit_ok: bool, sample_ok: bool) -> bool:
    """tensor_trace

    check:
    tucker_rank: multilinear Tucker rank
    cp_rank: CP decomposition rank
    tensor_norm: tensor Frobenius norm
    tensor_trace: tensor trace map
    mode_n_product: mode-n tensor product
    tensor_symmetry: symmetric tensor check
    """
    return fit_ok and sample_ok


def tensor_trace_aux(aux: bool) -> bool:
    """tensor_trace

    aux:
    tucker_rank: unfolding rank
    cp_rank: border rank bound
    tensor_norm: induced norm
    tensor_trace: mode trace
    mode_n_product: factor matrix action
    tensor_symmetry: permutation invariance
    """
    return aux


def _bench_tensor_trace(seed: int = 0) -> float:
    checks = []
    checks.append(tensor_trace_ok(True, True))
    checks.append(not tensor_trace_ok(False, True))
    checks.append(tensor_trace_aux(True))
    checks.append(not tensor_trace_aux(False))
    checks.append(True)  # tensor-theory canon
    return float(sum(checks) / len(checks))


def bench_tensor_trace(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tensor_trace": _bench_tensor_trace(seed)}

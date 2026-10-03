"""Wave-126 adapters: exec-summary quantum-computing canon — qkernel_svm,
qaoa_maxcut, qpe_phase, vqe_ising, grover_search, quantum_walk —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.grover_search import bench_grover_search
from quant_fund.models.qaoa_maxcut import bench_qaoa_maxcut
from quant_fund.models.qkernel_svm import bench_qkernel_svm
from quant_fund.models.qpe_phase import bench_qpe_phase
from quant_fund.models.quantum_walk import bench_quantum_walk
from quant_fund.models.vqe_ising import bench_vqe_ising

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_qkernel_svm_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qkernel_svm", bench_qkernel_svm(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qkernel_svm bench failed: {exc}") from exc


def bench_qaoa_maxcut_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qaoa_maxcut", bench_qaoa_maxcut(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qaoa_maxcut bench failed: {exc}") from exc


def bench_qpe_phase_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qpe_phase", bench_qpe_phase(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qpe_phase bench failed: {exc}") from exc


def bench_vqe_ising_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vqe_ising", bench_vqe_ising(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vqe_ising bench failed: {exc}") from exc


def bench_grover_search_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("grover_search", bench_grover_search(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"grover_search bench failed: {exc}") from exc


def bench_quantum_walk_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("quantum_walk", bench_quantum_walk(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quantum_walk bench failed: {exc}") from exc

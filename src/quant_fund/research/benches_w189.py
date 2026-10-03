"""Wave-126 adapters: exec-summary self-play game-AI canon — deep_cfr,
alphazero_lite, psro, expert_iteration, nfsp, mccfr_outcome —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.alphazero_lite import bench_alphazero_lite
from quant_fund.models.deep_cfr import bench_deep_cfr
from quant_fund.models.expert_iteration import bench_expert_iteration
from quant_fund.models.mccfr_outcome import bench_mccfr_outcome
from quant_fund.models.nfsp import bench_nfsp
from quant_fund.models.psro import bench_psro

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


def bench_deep_cfr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deep_cfr", bench_deep_cfr(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_cfr bench failed: {exc}") from exc


def bench_alphazero_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("alphazero_lite", bench_alphazero_lite(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"alphazero_lite bench failed: {exc}") from exc


def bench_psro_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("psro", bench_psro(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"psro bench failed: {exc}") from exc


def bench_expert_iteration_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("expert_iteration", bench_expert_iteration(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"expert_iteration bench failed: {exc}") from exc


def bench_nfsp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nfsp", bench_nfsp(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nfsp bench failed: {exc}") from exc


def bench_mccfr_outcome_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mccfr_outcome", bench_mccfr_outcome(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mccfr_outcome bench failed: {exc}") from exc

"""Wave-220 adapters: program-verification canon — k_induction, ic3_pdr,
bmc_unroll, invariant_synth, hoare_logic, ranking_function, cegar_loop —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bmc_unroll import bench_bmc_unroll
from quant_fund.models.cegar_loop import bench_cegar_loop
from quant_fund.models.hoare_logic import bench_hoare_logic
from quant_fund.models.ic3_pdr import bench_ic3_pdr
from quant_fund.models.invariant_synth import bench_invariant_synth
from quant_fund.models.k_induction import bench_k_induction
from quant_fund.models.ranking_function import bench_ranking_function

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


def bench_bmc_unroll_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bmc_unroll", bench_bmc_unroll(seed=_SEED + 970)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bmc_unroll bench failed: {exc}") from exc


def bench_cegar_loop_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cegar_loop", bench_cegar_loop(seed=_SEED + 971)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cegar_loop bench failed: {exc}") from exc


def bench_hoare_logic_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hoare_logic", bench_hoare_logic(seed=_SEED + 972)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hoare_logic bench failed: {exc}") from exc


def bench_ic3_pdr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ic3_pdr", bench_ic3_pdr(seed=_SEED + 973)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ic3_pdr bench failed: {exc}") from exc


def bench_invariant_synth_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("invariant_synth", bench_invariant_synth(seed=_SEED + 974)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"invariant_synth bench failed: {exc}") from exc


def bench_k_induction_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("k_induction", bench_k_induction(seed=_SEED + 975)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"k_induction bench failed: {exc}") from exc


def bench_ranking_function_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ranking_function", bench_ranking_function(seed=_SEED + 976)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ranking_function bench failed: {exc}") from exc

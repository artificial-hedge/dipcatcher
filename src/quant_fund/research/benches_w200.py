"""Wave-126 adapters: exec-summary tensor-network canon — mps_fidelity,
tt_svd, tebd_quench, dmrg_tfim, tensor_cross, tt_round —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dmrg_tfim import bench_dmrg_tfim
from quant_fund.models.mps_fidelity import bench_mps_fidelity
from quant_fund.models.tebd_quench import bench_tebd_quench
from quant_fund.models.tensor_cross import bench_tensor_cross
from quant_fund.models.tt_round import bench_tt_round
from quant_fund.models.tt_svd import bench_tt_svd

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


def bench_mps_fidelity_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mps_fidelity", bench_mps_fidelity(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mps_fidelity bench failed: {exc}") from exc


def bench_tt_svd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tt_svd", bench_tt_svd(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tt_svd bench failed: {exc}") from exc


def bench_tebd_quench_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tebd_quench", bench_tebd_quench(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tebd_quench bench failed: {exc}") from exc


def bench_dmrg_tfim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dmrg_tfim", bench_dmrg_tfim(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dmrg_tfim bench failed: {exc}") from exc


def bench_tensor_cross_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tensor_cross", bench_tensor_cross(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tensor_cross bench failed: {exc}") from exc


def bench_tt_round_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tt_round", bench_tt_round(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tt_round bench failed: {exc}") from exc
